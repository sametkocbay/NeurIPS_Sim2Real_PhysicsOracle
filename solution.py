"""PhysicsOracle, NeurIPS 2026 RealPDE Track 1: training package of the final phase.

train(data_dir, output_dir, device) reproduces our Codabench submission from the public training data:
  1. cache    the 32 x 64 training cache from data_dir/train_{real,sim}        (tools/make_cache.py)
              and the data tables of the gated decode, fitted on train_real   (tools/fit_gate.py)
  2. members  configs/final.yaml with seeds 3 and 4, EMA weights after epoch 59 (python -m rpde.train)
  3. compile  the two-member ensemble, ahead of time, batch sizes 1-5 and 10    (tools/aot_build_ens.py)
  4. package  the Codabench package layout in output_dir, clean-room tested     (tools/pack_submission.py)
load_predictor(output_dir, device) imports output_dir/submission.py, the runtime that ran on Codabench, unchanged.

Nothing is selected: the seeds (3, 4), the epoch (59), the decode settings and the compiled batch sizes are fixed in
configs/final.yaml (section `submission`), next to the training recipe.
RPDE_SMOKE=1 runs the same pipeline with one short epoch and one compiled group, to test the plumbing only.
"""
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "configs", "final.yaml")
_CFG = yaml.safe_load(open(CONFIG))
SUB = _CFG["submission"]                           # seeds, epoch, decode, gate coefficients, compiled batch sizes
assert int(SUB["epoch"]) == int(_CFG["eval"]["stop_after"]) == int(_CFG["eval"]["save_to"]), \
    "submission.epoch must equal eval.stop_after and eval.save_to"
SMOKE = os.environ.get("RPDE_SMOKE") == "1"


class _Log:
    def __init__(self, path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg):
        line = f"[solution {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")


class _GpuPeak(threading.Thread):
    """Device-level peak memory (all processes on the GPU, CUDA contexts included), polled through nvidia-smi."""

    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.every, self.peak_mib, self.ok = every, 0, shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()

    def run(self):
        while self.ok and not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                      "-i", "0"], capture_output=True, text=True, timeout=30).stdout
                self.peak_mib = max(self.peak_mib, int(out.split()[0]))
            except Exception:
                pass
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()


def _run(cmd, log, env, logfile):
    """Run a stage in a fresh interpreter, streaming its output to stdout and to `logfile`."""
    log("$ " + " ".join(cmd))
    t0 = time.time()
    with open(logfile, "a", buffering=1) as fh:
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"stage failed with exit code {rc}: {' '.join(cmd)}")
    return time.time() - t0


def train(data_dir, output_dir, device):
    t_start = time.time()
    out = os.path.abspath(output_dir)
    work = os.path.join(out, "work")
    logs = os.path.join(out, "logs")
    os.makedirs(work, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    log = _Log(os.path.join(logs, "solution.log"))
    stage_log = os.path.join(logs, "stages.log")
    cache, runs, aot = (os.path.join(work, d) for d in ("cache32", "runs", "aot"))
    env = dict(os.environ, RPDE_CACHE=cache, RPDE_RUNS=runs, PYTHONUNBUFFERED="1")
    if not env.get("CUDA_HOME") and os.path.isdir("/usr/local/cuda"):
        env["CUDA_HOME"] = "/usr/local/cuda"
    py = sys.executable
    gpu = _GpuPeak()
    gpu.start()
    times = {}
    log(f"data_dir={data_dir} output_dir={out} device={device} smoke={SMOKE}")

    # 1. cache: every second row and column of the raw fields (the evaluation grid), one npz per trajectory
    workers = str(min(16, os.cpu_count() or 1))
    times["cache_s"] = _run([py, "tools/make_cache.py", "--raw", data_dir, "--cache", cache, "--workers", workers],
                            log, env, stage_log)
    gate_json = os.path.join(logs, "gate.json")
    times["fit_gate_s"] = _run([py, "tools/fit_gate.py", "--out", gate_json, "--check"], log, env, stage_log)
    gate = json.load(open(gate_json))

    # 2. members, one after the other
    sets = []
    if SMOKE:
        sets = ["--set", "eval.stop_after=1", "--set", "eval.save_from=1", "--set", "eval.save_to=1",
                "--set", "data.windows_per_epoch=200"]
    epoch = 1 if SMOKE else int(SUB["epoch"])
    members = []
    for seed in SUB["seeds"]:
        name = f"fm_all_s{seed}"
        times[f"train_{name}_s"] = _run([py, "-m", "rpde.train", "--config", CONFIG, "--fold", "full",
                                          "--seed", str(seed), "--name", name, "--runs-dir", runs,
                                          "--device", str(device), *sets], log, env, stage_log)
        ck = os.path.join(runs, name, f"ep{epoch}.pth")
        if not os.path.isfile(ck):
            raise RuntimeError(f"missing snapshot {ck}")
        members.append(ck)
        for f in ("log.txt", "config.json", "run.json"):
            if os.path.isfile(os.path.join(runs, name, f)):
                os.makedirs(os.path.join(logs, name), exist_ok=True)
                shutil.copy(os.path.join(runs, name, f), os.path.join(logs, name, f))

    # 3. compile the ensemble for the batch sizes of the evaluation (traced on real training windows)
    samples = os.path.join(work, "sample_inputs.npy")
    _run([py, "tools/make_sample_inputs.py", "--cache", cache, "--out", samples], log, env, stage_log)
    groups = ("5,1",) if SMOKE else tuple(SUB["compile_batches"])
    dec = SUB["decode"]
    decode = [f"gamma={dec['gamma']}", f"band_scale={dec['band_scale']}",
              f"energy_preserve={int(bool(dec['energy_preserve']))}"]
    t0 = time.time()
    for g in groups:
        cmd = [py, "tools/aot_build_ens.py", aot, ",".join(members), *decode, f"gated={gate_json}",
               f"batches={g}", f"inputs={samples}"]
        if SMOKE:
            cmd.append("autotune=0")
        for attempt in (1, 2):
            try:
                _run(cmd, log, env, stage_log)
                break
            except RuntimeError as e:
                log(f"compile of batches {g} failed (attempt {attempt}): {e}")
    times["compile_s"] = time.time() - t0
    reports = {}
    for rp in sorted(glob.glob(os.path.join(aot, "build_report_*.json"))):
        shutil.copy(rp, logs)
        reports.update(json.load(open(rp))["runs"])
    log(f"compiled batch sizes: {sorted(int(b) for b in reports)} "
        f"(max |pred - eager| {max([r['max_dpred'] for r in reports.values()] or [float('nan')]):.2e})")

    # 4. package: the Codabench layout, written straight into output_dir; then predict() in a clean interpreter
    sys.path.insert(0, HERE)
    from tools.pack_submission import assemble, clean_room
    assemble(aot, members, out, ens_cfg=dict(gamma=float(dec["gamma"]), band_scale=float(dec["band_scale"]),
                                             energy_preserve=bool(dec["energy_preserve"]), gated=gate,
                                             mirror_views=False))
    clean_room(out)

    gpu.stop()
    summary = dict(times_s={k: round(v, 1) for k, v in times.items()}, total_s=round(time.time() - t_start, 1),
                   peak_gpu_mib_device=gpu.peak_mib if gpu.ok else None, compiled=reports, members=members,
                   smoke=SMOKE)
    json.dump(summary, open(os.path.join(logs, "summary.json"), "w"), indent=1)
    log(f"summary: {json.dumps({k: summary[k] for k in ('times_s', 'total_s', 'peak_gpu_mib_device')})}")
    if os.environ.get("RPDE_KEEP_WORK") != "1":
        shutil.rmtree(work, ignore_errors=True)
    log("done")


class Predictor:
    """The Codabench runtime (output_dir/submission.py): compiled graphs, eager fallback, warm-up at load time."""

    def __init__(self, output_dir, device):
        path = os.path.join(os.path.abspath(output_dir), "submission.py")
        spec = importlib.util.spec_from_file_location("physicsoracle_submission", path)
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)

    def predict(self, input_array):
        return self.runtime.predict(input_array, {})


def load_predictor(output_dir, device):
    return Predictor(output_dir, device)
