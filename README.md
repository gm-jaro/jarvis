# jarvis

Proof of concept: a small local LLM (Bielik 1.5B fine-tuned for tool calling, or Gemma 4) controls a pan-tilt
camera head through tool calls. A vision model describes what the camera sees, so the LLM can answer questions
like "what is on the left?" by turning the camera, looking and replying.

- `agent.py` - tool loop (`turn_camera`, `describe_view`)
- `vision.py` - USB camera and image description through Ollama
- `pantilt.py`, `firmware/` - serial driver and firmware for a Wemos D1 mini with two SG90 servos
- `server.py`, `web/` - web UI with live preview; `jarvis.py` is the terminal version
- `serve_bielik.sh`, `serve_ollama.sh` - model servers in Docker (vLLM / Ollama, AMD ROCm)
