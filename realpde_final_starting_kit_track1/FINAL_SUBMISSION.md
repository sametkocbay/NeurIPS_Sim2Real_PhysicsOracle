# Decision Phase: training package

Each invited team sends one training package for each track it is invited to. The organizers build the package, train your method from scratch on the server described below, and evaluate the trained model on the private test set. Training and evaluation both run inside your container with no network access.

## What to send

- Your image, built from your Dockerfile for linux/amd64 and pushed to Docker Hub. Send its name with the digest, for example `user/realpde-t1@sha256:...`. Keep the repository private and give the Docker Hub account `w3nhao` read access.
- A zip file with the same `Dockerfile`, `solution.py`, the rest of your code, a `README.md` and a `METHOD.md`. We read it to review the code, and rebuild from it if the image cannot be pulled.

By October 9, 2026, 24:00 UTC, email realpde-competition@googlegroups.com with the image name and digest, the Codabench submission id this package reproduces, and a Google Drive link to the zip.

## Interface

`solution.py` sits in `/workspace`, the working directory of the image. It defines the following functions.

- `train(data_dir, output_dir, device)`, in both tracks. It trains from scratch on the data in `data_dir` and writes everything inference needs into `output_dir`. `device` is `"cuda"`.
- Track 1: `load_predictor(output_dir, device)`. It returns an object with `predict(input_array)`. Input and output match Codabench: a float32 array of shape `(N, 20, 32, 64, 3)` in physical units, where N can be any batch size, and a prediction of the same shape or a dict with `prediction` and optional `lower` and `upper`.
- Track 2: `get_ttt_model(output_dir, device)`. It returns an object with `reset_ttt_state()` and `ttt_step(input_norm, prev_target_norm=None)`. The contract is the one used on Codabench, with `output_dir` in place of the submission directory. Inputs are normalized with the official `mean_std_real.pt`, as on Codabench.

Templates for both tracks are in the starting kit of the Final phase, each with a Dockerfile, a README template and a method template. They run as they are and train a small convolutional model, so you can check the interface end to end before porting your own method.

## How we run it

Training:

```
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/train_data:/data:ro -v /path/to/output:/output IMAGE \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

Evaluation uses the same image, also without network. Our script imports `solution` from `/workspace`, loads your model from `/output`, and runs it over the private test set on one A800.

## Data inside the container

`/data` is read-only. It holds the public training release as distributed: `train_sim/`, `train_real/` without `7575_0.h5`, and the official `mean_std_real.pt`. Nothing else is mounted. If your method starts from an official baseline checkpoint, include the file in your package and name it in the README. Every other checkpoint your method uses must be produced by `train()`.

## Our server

- CPU: 2 × Intel Xeon Platinum 8358, 64 cores and 128 threads in total
- GPU: 8 × NVIDIA A800-SXM4-80GB with NVLink, driver 550.163.01, CUDA 12.4
- Memory: 1 TB
- OS: Ubuntu 24.04 with Docker 29

By default each package gets one A800, 16 CPU cores and 128 GB of memory, and training has to finish within 24 hours. If your method needs more, ask for it in your submission email and give the reason. We review each request. Keep the CUDA version in your image at 12.4 or lower.

## README.md

Use the README template. It asks for the following.

- How to reproduce: build and train commands, if they differ from ours.
- The hardware your submitted model was trained on: GPU model and count, CPU and memory, with the training time and peak GPU memory on that hardware.
- The environment: base image, and every added package with its version, pinned the same way in the Dockerfile.
- The Codabench submission id this package reproduces.
- Seeds. If your method picks among several runs or checkpoints, the picking has to happen inside `train()` and use the training data only.
- Data and weights: confirm that training uses only the released training data, describe your augmentation, list any official checkpoints you start from, and list third-party code with its license.
- Track 2: whether your method calls the LLM gateway. If it does, the gateway stays reachable during your evaluation, as on Codabench, and nothing else is.
- A contact who can reply within 48 hours during the Decision Phase.

Please also attach the log of the training run behind your Codabench submission.

## METHOD.md

Describe the method in `METHOD.md`, following the method template: the model, the data you used and how, each training stage with its loss, optimizer, schedule and length, how you pick runs or checkpoints, how the uncertainty bounds are built, what happens at inference, what you did for speed, and which changes mattered most.
