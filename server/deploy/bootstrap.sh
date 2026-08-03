#!/bin/bash
# TensorDex serving bootstrap (plan Phase 4) — idempotent, run as root
# on an i4i-class box after cloning the repo to /home/ubuntu/TensorDex.
#
# Auth note: S3 access comes from the instance profile (read-only
# scope). No static keys anywhere in this script or the units.
set -euo pipefail

REPO=/home/ubuntu/TensorDex
NVME_DEV=${NVME_DEV:-/dev/nvme1n1}

# 1. Instance-store NVMe: xfs at /mnt/nvme. relatime, NOT noatime —
#    the evictor's LRU reads atime. Ephemeral by design; catalog and
#    manifests stay on the EBS root.
if ! mountpoint -q /mnt/nvme; then
    blkid "$NVME_DEV" >/dev/null 2>&1 || mkfs.xfs -q "$NVME_DEV"
    mkdir -p /mnt/nvme
    UUID=$(blkid -s UUID -o value "$NVME_DEV")
    grep -q "$UUID" /etc/fstab || \
        echo "UUID=$UUID /mnt/nvme xfs defaults,relatime,nofail 0 2" \
            >> /etc/fstab
    mount /mnt/nvme
fi
mkdir -p /mnt/nvme/cache /mnt/nvme/spill /srv/manifests "$REPO/logs"
chown ubuntu:ubuntu /mnt/nvme/cache /mnt/nvme/spill /srv/manifests \
    "$REPO/logs"

# 2. System packages + Python ≤3.12 venv (PyO3 0.20 wheel constraint).
apt-get update -qq
apt-get install -y -qq nginx build-essential pkg-config libssl-dev
sudo -u ubuntu bash -c '
    set -e
    cd '"$REPO"'
    command -v ~/.local/bin/uv >/dev/null || \
        curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v ~/.cargo/bin/rustc >/dev/null || \
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | \
            sh -s -- -y --profile minimal
    . ~/.cargo/env
    [ -d .venv ] || ~/.local/bin/uv venv --python 3.12 .venv
    ~/.local/bin/uv pip install -q -e . -e ./server boto3 psutil
'

# 3. Manifests (offline job; rerun at ingest/compaction).
sudo -u ubuntu "$REPO/.venv/bin/python" -m tensordex_serving.build_manifests \
    --db "$REPO/eval/raw/hub_compressed/metadata.db" \
    --out /srv/manifests --prefix compressed_eval

# 4. nginx front + systemd units.
cp "$REPO/server/deploy/nginx.conf" /etc/nginx/sites-available/td-cache
ln -sf /etc/nginx/sites-available/td-cache /etc/nginx/sites-enabled/td-cache
rm -f /etc/nginx/sites-enabled/default
nginx -t
cp "$REPO"/server/deploy/td-*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now nginx td-metadata td-materializer td-evictor
systemctl restart nginx

# 5. Smoke.
sleep 2
curl -sf localhost:8701/healthz >/dev/null && echo "metadata ok"
curl -sf localhost:8700/healthz >/dev/null && echo "cache front ok"
echo "bootstrap complete"
