"""Train the bank (A, B) and the victims (V1, V2, V3). 3 seeds each, 15 epochs,
SGD, a checkpoint saved after every epoch.
Run from the repository root: python -m frankenstein.train_bank [bank|victims|all]."""

import gc
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from . import config as C
from . import models as M


class DeviceLoader:
    """CIFAR-10 held as normalised tensors on the training device, with the usual random
    crop (pad 4) + horizontal flip done on the device. Same recipe as torchvision's
    RandomCrop/RandomHorizontalFlip, without the per-image CPU cost (which dominates on a
    laptop). Iterates (x, y) batches like a DataLoader."""

    def __init__(self, x, y, batch_size, train):
        self.x, self.y, self.bs, self.train = x, y, batch_size, train

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.train else -(-n // self.bs)

    def __iter__(self):
        n, dev = len(self.x), self.x.device
        if not self.train:
            for i in range(0, n, self.bs):
                yield self.x[i:i + self.bs], self.y[i:i + self.bs]
            return
        perm = torch.randperm(n, device=dev)
        ar = torch.arange(32, device=dev)
        for i in range(0, n - self.bs + 1, self.bs):       # drop_last
            idx = perm[i:i + self.bs]
            xb = torch.nn.functional.pad(self.x[idx], (4, 4, 4, 4))
            dy, dx = (torch.randint(0, 9, (self.bs,), device=dev) for _ in range(2))
            flip = torch.rand(self.bs, device=dev) < 0.5
            iy = (ar[None] + dy[:, None])[:, None, :, None]
            cols = torch.where(flip[:, None], 31 - ar[None], ar[None]) + dx[:, None]
            ix = cols[:, None, None, :]
            xb = xb[torch.arange(self.bs, device=dev)[:, None, None, None],
                    torch.arange(3, device=dev)[None, :, None, None], iy, ix]
            yield xb, self.y[idx]


def get_loaders(batch_size=C.BATCH_SIZE):
    norm = T.Compose([T.ToTensor(), T.Normalize(C.CIFAR_MEAN, C.CIFAR_STD)])

    def tensors(train):
        ds = torchvision.datasets.CIFAR10(C.DATA_DIR, train=train, download=True, transform=norm)
        x = torch.stack([ds[i][0] for i in range(len(ds))]).to(C.DEVICE)
        return x, torch.tensor(ds.targets).to(C.DEVICE)

    return (DeviceLoader(*tensors(True), batch_size, True),
            DeviceLoader(*tensors(False), 256, False))


def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        correct += (net(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_one(name, arch, seed, train_loader, test_loader):
    set_seed(seed)
    net = M.build_from_arch(arch).to(C.DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=C.LR, momentum=C.MOMENTUM,
                          weight_decay=C.WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.TRAIN_EPOCHS)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, C.TRAIN_EPOCHS + 1):
        net.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            opt.zero_grad(set_to_none=True)
            crit(net(x), y).backward()
            opt.step()
            if C.DEVICE == "mps" and step % 50 == 0:
                torch.mps.synchronize()     # the CPU otherwise queues kernels faster than the GPU runs them (memory blow-up)
        sched.step()
        path = os.path.join(C.CKPT_DIR, f"{name}_s{seed}_e{epoch}.pt")
        torch.save({"state_dict": net.state_dict(), "arch": arch,
                    "name": name, "seed": seed, "epoch": epoch}, path)
    acc = evaluate(net, test_loader)
    print(f"[{name} s{seed}] final test acc = {acc:.4f}")
    del net, opt, sched
    gc.collect()
    if C.DEVICE == "mps":
        torch.mps.empty_cache()     # between models only: the MPS allocator cache otherwise piles up (OOM)
    return acc


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    train_loader, test_loader = get_loaders()
    groups = {}
    if which in ("bank", "all"):
        groups.update(C.BANK_ARCHS)
    if which in ("victims", "all"):
        groups.update(C.VICTIM_ARCHS)
    log = {}
    for name, arch in groups.items():
        for seed in C.TRAIN_SEEDS:
            done = os.path.join(C.CKPT_DIR, f"{name}_s{seed}_e{C.TRAIN_EPOCHS}.pt")
            if os.path.exists(done):
                print(f"[{name} s{seed}] already trained, skipping")
                continue
            log[f"{name}_s{seed}"] = train_one(name, arch, seed, train_loader, test_loader)
    torch.save(log, os.path.join(C.CKPT_DIR, "_acc_log.pt"))
    print("done:", {k: round(v, 4) for k, v in log.items()})


if __name__ == "__main__":
    main()
