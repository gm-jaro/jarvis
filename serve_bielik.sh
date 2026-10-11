#!/usr/bin/env bash
# vLLM server (OpenAI API on 127.0.0.1:8000, model "bielik-<variant>") with Bielik-1.5B in a rocm/vllm container.
set -euo pipefail
J="$(cd "$(dirname "$0")" && pwd)"
IMG=rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0
VARIANT=${VARIANT:-dpo}
MODEL=${MODEL:-$(realpath -m "$J/../bielik-care/train/runs/${VARIANT}_1.5b_v1/merged")}
NAME=jarvis-bielik
CACHE="$J/.cache/vllm-docker"

case "${1:-}" in
  start)
    mkdir -p "$CACHE"/{home,triton,vllm}
    docker run -d --rm --name "$NAME" \
      --device=/dev/kfd --device=/dev/dri \
      --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
      --ipc=host --shm-size=4g --security-opt seccomp=unconfined \
      -e HOME=/cache/home -e XDG_CACHE_HOME=/cache -e TRITON_CACHE_DIR=/cache/triton \
      -e VLLM_CACHE_ROOT=/cache/vllm -e HF_HUB_OFFLINE=1 -e HIP_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
      -v "$MODEL":/model:ro -v "$CACHE":/cache -p 127.0.0.1:8000:8000 \
      --entrypoint python3 "$IMG" -m vllm.entrypoints.openai.api_server \
        --model /model --served-model-name "bielik-$VARIANT" --host 0.0.0.0 --port 8000 \
        --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.3 \
        --chat-template /model/chat_template_infer.jinja \
        --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
    ;;
  stop)
    docker exec "$NAME" chown -R "$(id -u):$(id -g)" /cache 2>/dev/null || true
    docker stop "$NAME"
    # --rm removes the container asynchronously; without waiting, a start run right away hits a taken name
    while docker container inspect "$NAME" >/dev/null 2>&1; do sleep 0.2; done
    ;;
  logs) docker logs -f "$NAME" ;;
  *) echo "usage: $0 start|stop|logs" >&2; exit 1 ;;
esac
