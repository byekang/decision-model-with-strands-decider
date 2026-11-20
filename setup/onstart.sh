#!/bin/bash
# Per-start setup for the workshop notebook instance. Run as root
# (nohup'd from the notebook lifecycle-config OnStart).
# The strands-decider E4B server runs on this instance's own GPU (:8004);
# notebooks talk to it via localhost — no cross-instance networking.
# Everything outside /home/ec2-user/SageMaker is wiped on stop/start, so the
# Jupyter kernel and its packages must be rebuilt here each time; the venv,
# model cache and repo live on the persistent volume (built by bootstrap.sh).
set -x
WORK=/home/ec2-user/SageMaker
PY=/home/ec2-user/anaconda3/envs/python3/bin
LOGS=$WORK/logs
mkdir -p $LOGS
chown ec2-user:ec2-user $LOGS
exec >> $LOGS/onstart.log 2>&1

echo "===== onstart $(date -u +%FT%TZ) ====="

# CloudFormation WaitCondition signal: lets the stack wait until the decision
# server is actually serving. DECIDER_WAIT_URL (presigned URL) is injected by
# the lifecycle config; when unset this is a no-op. Re-signaling an already
# completed WaitCondition on restart is harmlessly ignored.
signal_cfn() {
  [ -n "$DECIDER_WAIT_URL" ] || return 0
  curl -s -X PUT -H 'Content-Type:' --data-binary \
    "{\"Status\":\"$1\",\"Reason\":\"$2\",\"UniqueId\":\"decider\",\"Data\":\"$1\"}" \
    "$DECIDER_WAIT_URL" || true
}

# graphviz + Korean fonts (graph captions; prevents jamo-decomposed rendering)
yum install -y -q graphviz || true
yum install -y -q google-noto-sans-cjk-ttc-fonts || yum install -y -q google-noto-sans-cjk-fonts || true
fc-cache -f || true

# workshop kernel. anthropic is version-pinned: newer SDKs send an API version
# header that Bedrock rejects ("Invalid API version" 400).
su ec2-user -c "$PY/pip install -q ipykernel 'anthropic==1.8.0' 'langgraph==1.2.18' boto3 graphviz pillow"
su ec2-user -c "$PY/python -m ipykernel install --user --name decision-ws \
  --display-name 'Python (decision-ws)' \
  --env STRANDS_HOST localhost --env STRANDS_PORT 8004"
echo "export STRANDS_HOST=localhost" >  /etc/profile.d/dmsd.sh
echo "export STRANDS_PORT=8004"      >> /etc/profile.d/dmsd.sh

# laya (notebook 06): large CPU-torch install, run in background
nohup su ec2-user -c "$PY/pip install -q torch --index-url https://download.pytorch.org/whl/cpu && $PY/pip install -q laya" > $LOGS/laya-install.log 2>&1 &

# wait for the one-time bootstrap (first boot: venv/model still downloading)
for i in $(seq 1 240); do [ -f $WORK/.dmsd_bootstrap_done ] && break; sleep 15; done
[ -f $WORK/.dmsd_bootstrap_done ] || {
  echo ONSTART_BOOTSTRAP_TIMEOUT
  signal_cfn FAILURE "bootstrap (venv/model download) did not finish"
  exit 1
}

# strands-decider E4B on the GPU (:8004, localhost only)
su ec2-user -c "cd $WORK && HF_HOME=$WORK/hf-cache \
  nohup $WORK/venvs/decider/bin/strands-decider serve \
  StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --host 127.0.0.1 --port 8004 > $LOGS/decider.log 2>&1 &"

for i in $(seq 1 90); do curl -sf http://localhost:8004/health >/dev/null && break; sleep 10; done
if curl -sf http://localhost:8004/health >/dev/null; then
  echo ONSTART_DECIDER_OK
  # in-env self test: one real Korean /v1/systemone decision, logged for
  # remote verification (notebook instances have no SSM access).
  curl -s -m 60 -X POST http://localhost:8004/v1/systemone \
    -H 'Content-Type: application/json' \
    -d '{"state":"고객 문의: 해외 송금 수수료가 궁금합니다.","model":"strands-decider-E4B-gemma4-v1-2610","questions":{"d1":{"type":"choice","instructions":"이 문의를 계속 처리해도 되는 안전한 문의인가?","criteria":{"safe":null,"blocked":null}}}}' \
    && echo "" && echo ONSTART_SELFTEST_DONE
  signal_cfn SUCCESS "strands-decider E4B healthy"
else
  echo ONSTART_DECIDER_FAIL
  tail -n 40 $LOGS/decider.log || true
  signal_cfn FAILURE "strands-decider server did not become healthy"
fi
echo "===== onstart done $(date -u +%FT%TZ) ====="
