#!/usr/bin/env bash
# Jarvis's own Ollama on 127.0.0.1:11435, with models in jarvis/.ollama.
set -euo pipefail
J="$(cd "$(dirname "$0")" && pwd)"
IMG=ollama/ollama:0.34.4-rocm  # pinned version: the "rocm" tag stays the one document_tagger uses
NAME=jarvis-ollama

give_back() {  # Ollama in the container writes as root; we give the files back to the user
  docker exec "$NAME" chown -R "$(id -u):$(id -g)" /root/.ollama 2>/dev/null || true
}

case "${1:-}" in
  start)
    mkdir -p "$J/.ollama"
    # No HSA_OVERRIDE_GFX_VERSION: gfx1100 is supported natively, and the override breaks the gfx1036 iGPU.
    docker run -d --rm --name "$NAME" \
      --device=/dev/kfd --device=/dev/dri \
      --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
      --security-opt seccomp=unconfined \
      -e HIP_VISIBLE_DEVICES=0 -e OLLAMA_HOST=0.0.0.0:11434 -e OLLAMA_KEEP_ALIVE=2m \
      -v "$J/.ollama":/root/.ollama -p 127.0.0.1:11435:11434 \
      "$IMG"
    ;;
  pull)
    docker exec "$NAME" ollama pull "${2:?give a model, e.g. gemma4:e4b}"
    give_back
    ;;
  stop)
    give_back
    docker stop "$NAME"
    while docker container inspect "$NAME" >/dev/null 2>&1; do sleep 0.2; done
    ;;
  logs) docker logs -f "$NAME" ;;
  *) echo "usage: $0 start|stop|logs|pull <model>" >&2; exit 1 ;;
esac
