"""Train the bank (A, B) and the victims (V1, V2, V3). 3 seeds each, 15 epochs,
SGD, a checkpoint saved after every epoch.
Run from the repository root: python -m frankenstein.train_bank [bank|victims|all]."""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from . import config as C
from . import models as M


def get_loaders(batch_size=C.BATCH_SIZE):
    norm = [T.ToTensor(), T.Normalize(C.CIFAR_MEAN, C.CIFAR_STD)]
    train = torchvision.datasets.CIFAR10(
        C.DATA_DIR, train=True, download=True,
        transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] + norm))
    test = torchvision.datasets.CIFAR10(C.DATA_DIR, train=False, download=True,
                                        transform=T.Compose(norm))
    tl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                     num_workers=4, drop_last=True)
    vl = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=4)
    return tl, vl


def set_seed(s):
    torch.manual_seed(s)
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
        for x, y in train_loader:
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            opt.zero_grad(set_to_none=True)
            crit(net(x), y).backward()
            opt.step()
        sched.step()
        path = os.path.join(C.CKPT_DIR, f"{name}_s{seed}_e{epoch}.pt")
        torch.save({"state_dict": net.state_dict(), "arch": arch,
                    "name": name, "seed": seed, "epoch": epoch}, path)
    acc = evaluate(net, test_loader)
    print(f"[{name} s{seed}] final test acc = {acc:.4f}")
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
