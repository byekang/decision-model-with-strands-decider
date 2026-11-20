#!/bin/bash
# One-time environment build for the workshop notebook instance
# (ml.g5.2xlarge: strands-decider serves the E4B decision model on the local
# GPU — no separate server instance; see the CloudFormation stack).
# Run as root (nohup'd from the notebook lifecycle-config OnCreate).
# Downloads ~16.5GB of model weights; takes ~10-15 min on first boot.
# Everything lands under /home/ec2-user/SageMaker so it survives stop/start.
set -x
WORK=/home/ec2-user/SageMaker
PY=/home/ec2-user/anaconda3/envs/python3/bin
LOGS=$WORK/logs
mkdir -p $LOGS
chown ec2-user:ec2-user $LOGS
exec >> $LOGS/bootstrap.log 2>&1

echo "===== bootstrap start $(date -u +%FT%TZ) ====="

# venv: strands-decider server (GPU torch). Kept separate from the notebook
# kernel so server and kernel dependencies never conflict.
su ec2-user -c "$PY/python -m venv $WORK/venvs/decider"
su ec2-user -c "$WORK/venvs/decider/bin/pip install -q --upgrade pip"
su ec2-user -c "$WORK/venvs/decider/bin/pip install -q 'strands-decider==0.3.0'" || { echo BOOTSTRAP_DECIDER_PIP_FAIL; exit 1; }

# model weights into the persistent volume: LoRA adapter repo + gemma-4-E4B-it
# base torso (resolved from the adapter config). Neither repo is gated.
su ec2-user -c "HF_HOME=$WORK/hf-cache $WORK/venvs/decider/bin/python -c \"
from huggingface_hub import snapshot_download
snapshot_download('StrandsAgents/strands-decider-E4B-gemma4-v1-2610')
snapshot_download('google/gemma-4-E4B-it')
\"" || { echo BOOTSTRAP_HF_DOWNLOAD_FAIL; exit 1; }

su ec2-user -c "touch $WORK/.dmsd_bootstrap_done"
echo "===== bootstrap done $(date -u +%FT%TZ) ====="
