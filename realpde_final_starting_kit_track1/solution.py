import glob
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

T_IN = 20
T_OUT = 20
SUB = 2
STEPS = (2000, 1000)
BATCH = 16
SEED = 0


class Net(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3 * T_IN, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 3 * T_OUT, 3, padding=1),
        )

    def forward(self, x):
        n, t, h, w, c = x.shape
        z = x.permute(0, 1, 4, 2, 3).reshape(n, t * c, h, w)
        z = self.body(z).reshape(n, T_OUT, c, h, w).permute(0, 1, 3, 4, 2)
        return x[:, -1:] + z


def load_windows(folder, real):
    windows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.h5"))):
        with h5py.File(path, "r") as f:
            u = f["u"][:, ::SUB, ::SUB]
            v = f["v"][:, ::SUB, ::SUB]
            p = np.zeros_like(u) if real else f["p"][:, ::SUB, ::SUB]
        x = np.stack([u, v, p], axis=-1).astype(np.float32)
        for t in range(0, x.shape[0] - T_IN - T_OUT + 1, T_IN):
            windows.append(x[t:t + T_IN + T_OUT])
    return torch.from_numpy(np.stack(windows))


def train(data_dir, output_dir, device):
    torch.manual_seed(SEED)
    sim = load_windows(os.path.join(data_dir, "train_sim"), real=False)
    real = load_windows(os.path.join(data_dir, "train_real"), real=True)
    mean = real.mean(dim=(0, 1, 2, 3)).to(device)
    std = real.std(dim=(0, 1, 2, 3)).to(device)
    std = torch.where(std == 0, torch.ones_like(std), std)
    model = Net().to(device)
    for data, steps in zip((sim, real), STEPS):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(steps):
            w = (data[torch.randint(len(data), (BATCH,))].to(device) - mean) / std
            loss = ((model(w[:, :T_IN]) - w[:, T_IN:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu()},
               os.path.join(output_dir, "model.pth"))


class Predictor:
    def __init__(self, output_dir, device):
        ckpt = torch.load(os.path.join(output_dir, "model.pth"), map_location="cpu")
        self.device = device
        self.mean = ckpt["mean"].to(device)
        self.std = ckpt["std"].to(device)
        self.model = Net().to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, input_array):
        x = torch.as_tensor(np.asarray(input_array, dtype=np.float32), device=self.device)
        y = self.model((x - self.mean) / self.std) * self.std + self.mean
        y[..., 2] = 0
        return y.cpu().numpy()


def load_predictor(output_dir, device):
    return Predictor(output_dir, device)
