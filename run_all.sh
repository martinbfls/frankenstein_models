#!/usr/bin/env bash
# Full pipeline: wait for CIFAR -> train bank+victims -> E1 -> E2 -> E3 -> finalize.
# Idempotent: checkpoints and stats are cached, so re-running resumes.
set -e
cd "$(dirname "$0")"
LOG=run_all.log
exec > >(tee -a "$LOG") 2>&1
echo "=== run_all start $(date) ==="

# 1) wait for the CIFAR archive to finish downloading
TAR=data/cifar-10-python.tar.gz
EXPECT_MD5=c58f30108f718f92721af3b95e74349a
echo "[1] waiting for CIFAR download..."
while ! grep -q "EXIT 0" data/curl.log 2>/dev/null; do
  sleep 15
done
MD5=$(md5sum "$TAR" | cut -d' ' -f1)
echo "[1] download done, md5=$MD5 (expect $EXPECT_MD5)"

# 2) train bank + victims (checkpoint every epoch)
echo "[2] training bank + victims $(date)"
python train_bank.py all

# 3) experiments
echo "[3] E0 validity $(date)"; python e0_validity.py || true
echo "[3] E1 $(date)"; python e1.py
echo "[3] E2 $(date)"; python e2.py
echo "[3] E3 $(date)"; python e3.py
echo "[4] finalize $(date)"; python finalize.py
echo "=== run_all done $(date) ==="
