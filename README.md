# PhysicsOracle

Repository for Team PhysicsOracle's NeurIPS 2026 RealPDE Competition, Track 1 submission.

Samet Kocbay · Maximilian Leutschafft · Oscar Breiner  — Technical University of Munich

This repository is used to create the final submission consisting of a docker image and a .zip file. The `README.md` you are reading explains how the submission is created, how it should look like and how to test it. For the readme of the submission refer to [final/README.md](final/README.md). For details about the submitted model, architecture and training, see [final/METHOD.md](final/METHOD.md).

## Repository layout

```text
├── README.md                         # repository guide
├── solution.py                       # official training and prediction entry points
├── pyproject.toml                    # local Python environment
├── configs/
│   └── final.yaml                    # training recipe and submission settings
├── final/                            # source files for submission documentation
│   ├── Dockerfile                    # training image definition
│   ├── README.md                     # submission instructions
│   ├── METHOD.md                     # method description
│   └── figures/                      # SVGs referenced by METHOD.md
├── rpde/                             # training, model and inference implementation
│   ├── cores/axial.py                # the single transformer core implementation
│   └── tools/                        # shipped data, compilation and predictor-assembly tools
├── submission/
│   └── submission.py                 # predictor runtime copied into training outputs
├── tools/                            # repository-only packaging and figure tools
│   ├── make_final_package.py
│   ├── make_architecture_figure.py
│   ├── make_improvement_figure.py
│   └── figure_data/submissions.csv    # score history used by the improvement figure
├── third_party/realpde_kit/           # organizer scorer and provenance README
├── logs/
│   ├── wandb/                        # original training logs included in submission
│   └── reproduction_4090/            # local reproduction evidence, excluded from submission
├── weights/                          # saved members from the original submission
└── realpde_final_starting_kit_track1/  # organizer instructions and templates
```

## Submission repository layout

The generated `dist/final/package/` folder is the Docker build context. Its contents are copied into `/workspace` in the image and included under `physicsoracle_track1/` in the ZIP:

```text
/workspace/
├── Dockerfile
├── README.md                         # from final/README.md
├── METHOD.md                         # from final/METHOD.md
├── solution.py
├── configs/final.yaml
├── figures/
│   ├── architecture.svg
│   └── progress.svg
├── rpde/                             # model, training and inference code
│   ├── cores/axial.py
│   └── tools/                        # helpers called by train()
├── submission/submission.py
├── third_party/realpde_kit/
│   ├── scoring.py
│   └── README.md
└── training_logs_948641/
    ├── fm_all_s3/                    # output_log.txt, history.txt, config.yaml
    └── fm_all_s4/                    # output_log.txt, history.txt, config.yaml
```

Training data is mounted separately at `/data`; trained weights and compiled graphs are generated in `/output`. The top-level development `tools/`, saved `weights/` and reproduction logs are excluded from the package.

## Assemble the training submission

Run from the repository root:

```bash
python tools/make_final_package.py
docker build -t physicsoracle dist/final/package
```

The first command assembles `dist/final/package/` and creates `dist/final/physicsoracle_track1.zip`. The second builds the Docker image from that same folder. The packaging script does not build or upload the image.

The Dockerfile copies the assembled folder into `/workspace`. Training reads the mounted `/data` directory and writes inference artifacts into the mounted `/output` directory. Follow [final/README.md](final/README.md#how-to-reproduce) for the container training command and the [organizer instructions](realpde_final_starting_kit_track1/FINAL_SUBMISSION.md) for delivery requirements.

Edit source files in this repository, then rerun the packaging command before building the image. Changes to source files do not automatically update an existing package or ZIP.

## Test the image

Running the image needs Docker with the NVIDIA Container Toolkit (`--gpus`). `/path/to/train_data` holds `train_real/` (without `7575_0.h5`) and `train_sim/`, as the organizers mount it. A smoke run checks the whole pipeline in about 5 minutes; then run the organizers' command unchanged:

```bash
docker run --rm --network none --gpus '"device=0"' --shm-size 32g -e RPDE_SMOKE=1 \
  -v /path/to/train_data:/data:ro -v /tmp/po_smoke:/output physicsoracle \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
grep -E "WARNING|compiled batch sizes|clean-room" /tmp/po_smoke/logs/solution.log   # expect [1, 5], no WARNING

docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/train_data:/data:ro -v /path/to/output:/output physicsoracle \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

The full run must log `compiled batch sizes: [1, 2, 3, 4, 5, 10]` and no `WARNING`. Use a fresh output directory for each run.

## Deliver the package

[FINAL_SUBMISSION.md](realpde_final_starting_kit_track1/FINAL_SUBMISSION.md) asks for the following by October 9, 2026, 24:00 UTC:

1. Push the image to a **private** Docker Hub repository and give the Docker Hub account `w3nhao` read access:
   ```bash
   docker tag physicsoracle <dockerhub-user>/physicsoracle-t1:final
   docker push <dockerhub-user>/physicsoracle-t1:final
   docker inspect --format '{{index .RepoDigests 0}}' <dockerhub-user>/physicsoracle-t1:final   # name@sha256:...
   ```
2. Upload `dist/final/physicsoracle_track1.zip`, built from the same package folder as the image, to Google Drive and create a share link.
3. Email realpde-competition@googlegroups.com with the team name (PhysicsOracle), the Codabench username (`physicsoracle`, as in final/README.md), the track (Track 1: Simulation-to-Real), the image name and digest (`<dockerhub-user>/physicsoracle-t1@sha256:...`), the Codabench submission id `948641`, and the Google Drive link. This list follows the organizers' email of 29 September 2026, which asks for more fields than FINAL_SUBMISSION.md.

## Work on training locally

Use Python 3.10. If using `uv`, install the dependencies declared in `pyproject.toml`:

```bash
uv sync
source .venv/bin/activate
```

Prepare the released data, whose parent directory contains `train_real/` and `train_sim/`, then train individual members:

```bash
python rpde/tools/make_cache.py --raw /path/to/train_data
python -m rpde.train --config configs/final.yaml --fold full --seed 3 --name fm_all_s3
python -m rpde.train --config configs/final.yaml --fold full --seed 4 --name fm_all_s4
```

The default cache is `data/cache32/`; set `RPDE_CACHE` to use another location. Training writes to `runs/<name>/`, including `ep59.pth`, `log.txt` and the resolved configuration. These commands train members only. The full `solution.train()` pipeline also fits the gate tables, compiles the ensemble and assembles the predictor. Use the Docker environment for the pinned training and CUDA compilation toolchain.

## Code entry points

| File | Responsibility |
|---|---|
| `solution.py` | `train(data_dir, output_dir, device)` and `load_predictor(output_dir, device)` |
| `rpde/train.py`, `rpde/data.py`, `rpde/losses.py` | training loop, data sampling and losses |
| `rpde/scaffold.py`, `rpde/cores/axial.py` | model implementation |
| `rpde/ensemble_predict.py`, `rpde/gated_decode.py`, `rpde/decode.py` | ensemble and output decoding |
| `rpde/tools/aot_build_ens.py` | compile the trained ensemble |
| `rpde/tools/pack_submission.py` | assemble the inference artifacts used after training |
| `tools/make_final_package.py` | assemble the training package and ZIP |

The training package and inference artifacts are different outputs. `load_predictor()` loads the generated `/output` artifacts; keep its weights, `ens_cfg.json`, runtime code and compiled graphs together.

## Update the documentation and figures

Keep submission instructions in `final/README.md` and the method description in `final/METHOD.md`.

Figure generation requires Matplotlib separately from the training dependencies:

```bash
python tools/make_architecture_figure.py
python tools/make_improvement_figure.py --output-dir final/figures --font-scale 1.15 --theme light
```

The architecture command writes `final/figures/architecture.svg`. The improvement command writes `final/figures/progress.svg` from `tools/figure_data/submissions.csv`. Inspect regenerated figures before packaging them.
