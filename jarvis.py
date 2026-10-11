"""Jarvis in the terminal: the brain model turns the camera through tools, the eyes model describes the view."""

import argparse

from agent import Agent, Head
from models import BRAINS, VISIONS

DIM, RESET = "\033[2m", "\033[0m"


def console_emit(kind, /, **d):
    # Brain content (kind="content") is skipped: tool calls and the answer arrive as separate events.
    if kind == "brain_token" and d["kind"] == "think":
        print(f"{DIM}{d['text']}{RESET}", end="", flush=True)
    elif kind == "brain_start":
        print(f"{DIM}[{d['model']}]{RESET}")
    elif kind == "brain_end":
        print(f"{DIM}\n[{d['tokens']} tok, {d['seconds']} s]{RESET}")
    elif kind == "tool_call":
        print(f"  > {d['label']}")
    elif kind == "vision_start":
        print(f"  {d['model']}> ", end="", flush=True)
    elif kind == "vision_token":
        print(d["text"], end="", flush=True)
    elif kind == "vision_end":
        print(f" {DIM}[{d['seconds']} s]{RESET}")
    elif kind == "tool_result" and d["name"] != "describe_view":
        print(f"    {d['result']}")
    elif kind == "answer":
        print(f"jarvis> {d['text']}")
    elif kind in ("notice", "error"):
        print(f"[{kind}] {d.get('text') or d.get('message')}")


def main():
    ap = argparse.ArgumentParser(description="Jarvis: the brain controls the camera, the eyes describe the image.")
    ap.add_argument("prompt", nargs="*", help="a single command; without it, a conversation in the terminal")
    ap.add_argument("--no-think", action="store_true", help="no reasoning: faster but weaker")
    ap.add_argument("--brain", choices=BRAINS, help="model that controls the camera")
    ap.add_argument("--vision", choices=VISIONS, help="model that describes the photo")
    a = ap.parse_args()

    head = Head()
    agent = Agent(head, console_emit)
    if a.brain or a.vision:
        agent.set_config(a.brain, a.vision)
    think = not a.no_think
    try:
        if a.prompt:
            agent.ask(" ".join(a.prompt), think)
            return
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if text in ("/exit", "/quit"):
                break
            if text == "/reset":  # Bielik's context is 8192 tokens
                agent.reset()
                head.look("center")
            elif text:
                agent.ask(text, think)
    except KeyboardInterrupt:
        pass
    finally:
        head.close()


if __name__ == "__main__":
    main()
