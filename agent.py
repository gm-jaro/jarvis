"""Jarvis: camera head and the brain model's tool loop. Used by jarvis.py (terminal) and server.py (web page)."""

import json
import re
import threading
import time
import unicodedata
from pathlib import Path

import httpx

from models import BRAINS, VISIONS, capabilities, load_config, save_config, unload
from pantilt import PanTilt
from vision import Camera, describe

MAX_MODEL_LEN = 8192  # --max-model-len in serve_bielik.sh
MAX_STEPS = 12
SETTLE_S = 0.3  # the image still shakes this long after the servos stop

SYSTEM = (
    "Jesteś Jarvis, asystent z kamerą na obrotowej głowicy. Masz dwa narzędzia: describe_view robi zdjęcie tego, "
    "na co kamera teraz patrzy, i zwraca opis, a turn_camera tylko obraca kamerę i niczego nie pokazuje. "
    "Nic nie widzisz, dopóki nie wywołasz describe_view, więc na każde pytanie o to, co widać, wywołaj "
    "describe_view, zamiast prosić o szczegóły.\n"
    "Przykład 1: użytkownik pyta „Czy ktoś jest przed kamerą?”, ty od razu wywołujesz describe_view "
    "z question=„Czy ktoś jest przed kamerą?”, a potem odpowiadasz na podstawie opisu.\n"
    "Przykład 2: użytkownik pyta „Co jest po lewej?”, ty wywołujesz turn_camera z direction=left, "
    "potem describe_view, a potem odpowiadasz na podstawie opisu.\n"
    "Odpowiadaj po polsku i opieraj się tylko na opisach z kamery."
)

# When the brain wants to answer without a single describe_view in this turn (asks back or makes up the view),
# it gets this reminder once; it still decides on its own whether to call the tool.
GROUNDING_NUDGE = (
    "Nie wywołałeś jeszcze describe_view, więc nie wiesz, co widzi kamera. Wywołaj teraz describe_view "
    "(pytanie użytkownika możesz przekazać w question), a potem odpowiedz na podstawie opisu."
)

# Direction values are in English: tool JSON was not translated in training and the model picks "left" itself.
# Named turn_camera, not look: with "look" Bielik 1.5B assumed the tool returns an image and turned in circles.
TOOLS = [
    {"type": "function", "function": {
        "name": "turn_camera",
        "description": "Obraca kamerę w podanym kierunku o podaną liczbę stopni (domyślnie 30); center ustawia "
                       "ją na wprost. Niczego nie pokazuje: zwraca tylko, gdzie kamera teraz patrzy.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["left", "right", "up", "down", "center"],
                          "description": "kierunek obrotu"},
            "degrees": {"type": "integer", "description": "o ile stopni obrócić, od 10 do 90"},
        }, "required": ["direction"]},
    }},
    {"type": "function", "function": {
        "name": "describe_view",
        "description": "Robi zdjęcie tego, na co kamera teraz patrzy, i zwraca opis. Nie wymaga obracania "
                       "kamery. Użyj go przy każdym pytaniu o to, co widać; w question możesz przekazać pytanie.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "opcjonalne pytanie o szczegół na obrazie"},
        }},
    }},
]

DIRECTIONS = {
    "left": "left", "lewo": "left", "right": "right", "prawo": "right",
    "up": "up", "gora": "up", "down": "down", "dol": "down",
    "center": "center", "srodek": "center", "prosto": "center",
}
CAMERA_TOOLS = ("turn_camera", "look")  # "look" is the old name, in case the model uses it
DIRECTION_PL = {"left": "to the left", "right": "to the right", "up": "up", "down": "down"}
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)
# </myśl>: in part of the SFT data the translator translated </think>
SPECIAL_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>|</?s>|<think>|</?myśl>")
LIMITS_RE = re.compile(r"P=(\d+)\.\.(\d+) T=(\d+)\.\.(\d+)")


def ascii_fold(text):
    text = text.lower().replace("ł", "l")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().strip()


def norm_degrees(degrees):
    try:
        degrees = int(degrees)
    except (TypeError, ValueError):
        degrees = 0
    return 30 if degrees <= 0 else max(10, min(90, degrees))


def decode_args(arguments):
    """Arguments as a dict."""
    args = arguments
    for _ in range(2):
        if not isinstance(args, str):
            break
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def parse_text_calls(content, step):
    """<tool_call>{...}</tool_call> calls from the model's reply and the text without them."""
    calls = []
    for i, raw in enumerate(TOOL_CALL_RE.findall(content)):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            calls.append((f"call_{step}_{i}", str(obj.get("name", "")), decode_args(obj.get("arguments", {}))))
    return calls, TOOL_CALL_RE.sub("", content).strip()


def split_think(text, think):
    """(reasoning, content)."""
    if not think:
        return "", text
    reasoning, sep, content = text.partition("</think>")
    return (reasoning, content) if sep else (text, None)


def call_label(name, args):
    """Human-readable description of a tool call."""
    if name in CAMERA_TOOLS:
        kind = DIRECTIONS.get(ascii_fold(str(args.get("direction", "center"))))
        if kind == "center":
            return "Pointing the camera straight ahead"
        if kind:
            return f"Turning the camera {DIRECTION_PL[kind]} by {norm_degrees(args.get('degrees'))}°"
        return f"Unknown direction: {args.get('direction')!r}"
    if name == "describe_view":
        question = args.get("question")
        return "Asking Gemma to describe the view" + (f": “{question}”" if question else "")
    return f"Unknown tool {name!r}"


class Head:
    """Camera head; left/right/up/down directions are translated to servo angles using calibration.json."""

    def __init__(self, on_move=None):
        cal_path = Path(__file__).with_name("calibration.json")
        cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        self.pan_sign = 1 if cal.get("pan_right_increases", True) else -1
        self.tilt_sign = 1 if cal.get("tilt_up_increases", True) else -1
        self.on_move = on_move or (lambda state: None)
        self.lock = threading.Lock()
        self.pt = PanTilt()
        self.cam = Camera()
        self.limits = tuple(int(x) for x in LIMITS_RE.search(self.pt.command("L")).groups())
        self._set(self.pt.center())

    def _set(self, pos):
        self.pan, self.tilt = pos
        self.moved_at = time.monotonic()
        self.on_move(self.state())

    def _right(self, pan):
        return (pan - 90) * self.pan_sign

    def _up(self, tilt):
        return (tilt - 90) * self.tilt_sign

    def state(self):
        pmin, pmax, tmin, tmax = self.limits
        return {
            "pan": self.pan, "tilt": self.tilt, "where": self.where(),
            "right": self._right(self.pan), "up": self._up(self.tilt),
            "right_range": sorted((self._right(pmin), self._right(pmax))),
            "up_range": sorted((self._up(tmin), self._up(tmax))),
        }

    def where(self):
        right, up = self._right(self.pan), self._up(self.tilt)
        parts = []
        if right:
            parts.append(f"{abs(right)}° w {'prawo' if right > 0 else 'lewo'}")
        if up:
            parts.append(f"{abs(up)}° w {'górę' if up > 0 else 'dół'}")
        return ", ".join(parts) or "na wprost"

    def look(self, direction="center", degrees=30):
        kind = DIRECTIONS.get(ascii_fold(str(direction)))
        if kind is None:
            return f"Nieznany kierunek {direction!r}. Dostępne: left, right, up, down, center."
        degrees = norm_degrees(degrees)
        with self.lock:
            pan, tilt = self.pan, self.tilt
            if kind == "center":
                pan, tilt = 90, 90
            elif kind in ("left", "right"):
                pan += degrees * self.pan_sign * (1 if kind == "right" else -1)
            else:
                tilt += degrees * self.tilt_sign * (1 if kind == "up" else -1)
            self._set(self.pt.move(pan, tilt))
        note = "" if (self.pan, self.tilt) == (pan, tilt) else " Dalej w tę stronę głowica się nie obróci."
        # Without this hint Bielik 1.5B often answers with just the angle, without looking at what is there.
        return f"Kamera patrzy teraz: {self.where()}.{note} Żeby zobaczyć, co tam jest, wywołaj describe_view."

    def close(self):
        try:
            with self.lock:
                self.pt.center()
        finally:
            self.cam.close()
            self.pt.close()


def vllm_label(entry, model_id):
    return f"{entry['label']} {model_id.removeprefix('bielik-').upper()}"


def to_ollama(messages):
    """History from OpenAI format (vLLM) to Ollama's /api/chat: arguments as a dict, result with the tool name."""
    out, names = [], {}
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                names[tc["id"]] = tc["function"]["name"]
                calls.append({"function": {"name": tc["function"]["name"],
                                           "arguments": decode_args(tc["function"]["arguments"])}})
            out.append({"role": "assistant", "content": m["content"], "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_name": names.get(m["tool_call_id"], "")})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class Agent:
    """Conversation with the selected "brain": streams its replies and runs the tool calls."""

    def __init__(self, head, emit):
        self.head, self.emit = head, emit
        self.http = httpx.Client(timeout=httpx.Timeout(300, connect=5))
        self.config = load_config()
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.busy = False
        self.cancelled = False
        self._start_lock = threading.Lock()

    @property
    def brain(self):
        return BRAINS[self.config["brain"]]

    @property
    def vision(self):
        return VISIONS[self.config["vision"]]

    def config_info(self):
        """Selected models and the lists to choose from, for the web page."""
        return {**self.config,
                "brains": {k: self._label(v) for k, v in BRAINS.items()},
                "visions": {k: v["label"] for k, v in VISIONS.items()}}

    def _label(self, entry):
        """Display name; for vLLM with the checkpoint variant, e.g. "Bielik 1.5B DPO"."""
        if entry.get("backend") != "vllm":
            return entry["label"]
        try:
            return vllm_label(entry, self._vllm_model_id(entry))
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            return entry["label"]

    def _vllm_model_id(self, entry):
        r = self.http.get(f"{entry['base']}/v1/models", timeout=2)
        r.raise_for_status()
        return r.json()["data"][0]["id"]  # bielik-dpo or bielik-sft (VARIANT in serve_bielik.sh)

    def set_config(self, brain=None, vision=None):
        """Switches models; changing the brain starts a new conversation. Frees unused Ollama models from VRAM."""
        new = {"brain": brain or self.config["brain"], "vision": vision or self.config["vision"]}
        if new["brain"] not in BRAINS or new["vision"] not in VISIONS:
            raise ValueError(f"unknown model: {new}")
        before = self._ollama_models()
        brain_changed = new["brain"] != self.config["brain"]
        self.config = new
        save_config(new)
        for base, model in before - self._ollama_models():
            threading.Thread(target=unload, args=(base, model), daemon=True).start()
        if brain_changed:
            self.reset()
        self.emit("config", **self.config_info())

    def _ollama_models(self):
        return {(e["base"], e["model"]) for e in (self.brain, self.vision) if "model" in e}

    def reset(self):
        self.messages = self.messages[:1]
        self.emit("reset")

    def start(self, target, *args):
        """Runs target(*args) in the background as an agent task; False if the agent is already working."""
        with self._start_lock:
            if self.busy:
                return False
            self.busy, self.cancelled = True, False
        threading.Thread(target=self._task, args=(target, *args), daemon=True).start()
        return True

    def _task(self, target, *args):
        self.emit("status", busy=True)
        try:
            target(*args)
        except Exception as e:
            self.emit("error", message=f"{type(e).__name__}: {e}")
        finally:
            self.busy = False
            self.emit("status", busy=False)

    def ask(self, text, think=True):
        self.emit("user", text=text)
        self.messages.append({"role": "user", "content": text})
        retried = nudged = looked = False
        force_tool = True  # the first step always uses a tool, see _generate_vllm
        for step in range(MAX_STEPS):
            content, calls = self._generate(think, step, force_tool)
            force_tool = False
            if self.cancelled:
                self.emit("cancelled")
                return
            if not calls:  # Bielik writes <tool_call> in the text, Ollama returns the calls separately
                calls, content = parse_text_calls(content, step)
            if not calls:
                if not content and not retried:
                    retried = True
                    self.emit("notice", text="The model gave no answer, sampling again.")
                    continue
                self.messages.append({"role": "assistant", "content": content})
                if not looked and not nudged:
                    nudged = force_tool = True
                    self.messages.append({"role": "user", "content": GROUNDING_NUDGE})
                    self.emit("notice", text="The brain wanted to answer without looking through the camera; "
                                             "reminding it about describe_view.")
                    continue
                self.emit("answer", text=content or "(no answer)")
                return
            # The history gets already decoded arguments, so the model does not see double encoding in its context.
            self.messages.append({"role": "assistant", "content": content, "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                for cid, name, args in calls
            ]})
            for i, (cid, name, args) in enumerate(calls):
                self.emit("tool_call", id=cid, name=name, args=args, label=call_label(name, args))
                result = self._call_tool(cid, name, args)
                if name in CAMERA_TOOLS and i + 1 < len(calls) and calls[i + 1][1] in CAMERA_TOOLS:
                    # Several turns in one reply: without this the view between them would be lost.
                    result += " Widok: " + self._call_tool(f"{cid}_auto", "describe_view", {})
                looked = looked or name == "describe_view" or " Widok: " in result
                self.emit("tool_result", id=cid, name=name, result=result)
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": result})
        self.emit("answer", text=f"(stopping: more than {MAX_STEPS} tool steps)")

    def describe_now(self, call_id):
        """View description on request from the web page, without the brain."""
        self._describe(call_id)

    def _call_tool(self, cid, name, args):
        try:
            if name in CAMERA_TOOLS:
                return self.head.look(args.get("direction", "center"), args.get("degrees", 30))
            if name == "describe_view":
                return self._describe(cid, args.get("question"))
            return f"Nie ma narzędzia {name!r}. Dostępne: turn_camera, describe_view."
        except Exception as e:  # the model should see the error and try another way, not end the conversation
            return f"Błąd narzędzia {name}: {e}"

    def _describe(self, call_id, question=None):
        question = str(question).strip() if question else None
        vision = self.vision
        jpg, path = self.head.cam.snapshot(after=self.head.moved_at + SETTLE_S)
        where = self.head.where()
        self.emit("snapshot", id=call_id, url=f"/snapshots/{path.name}", where=where)
        self.emit("vision_start", id=call_id, question=question, model=vision["label"])
        t0 = time.monotonic()
        text = describe(jpg, vision, question, on_token=lambda s: self.emit("vision_token", id=call_id, text=s))
        self.emit("vision_end", id=call_id, seconds=round(time.monotonic() - t0, 1))
        return f"[kamera: {where}] {text}"

    def _generate(self, think, step, force_tool=False):
        """One brain reply, streamed through emit; returns (content, tool calls)."""
        brain = self.brain
        if brain["backend"] == "vllm":
            return self._generate_vllm(brain, think, step, force_tool), []
        return self._generate_ollama(brain, think, step)

    def _generate_vllm(self, brain, think, step, force_tool):
        """Bielik in vLLM."""
        base = brain["base"]
        model_id = self._vllm_model_id(brain)
        template_kwargs = {"enable_thinking": think}
        if force_tool:
            # With tools the bielik-tools template says to ask back when "parameters are missing", and Bielik
            # 1.5B asked for details on "Czy ktoś jest przed kamerą?" (2/6). The tool_choice == "required" branch
            # turns that into "You MUST invoke at least one tool"; only in selected steps, so it can answer later.
            template_kwargs["tool_choice"] = "required"
        body = {"model": model_id, "messages": self.messages, "tools": TOOLS,
                "chat_template_kwargs": template_kwargs}
        r = self.http.post(f"{base}/v1/chat/completions/render", json=body)
        r.raise_for_status()
        ids = r.json()["token_ids"]
        budget = MAX_MODEL_LEN - len(ids)
        if budget < 256:
            raise RuntimeError("Bielik's context is full (8192 tokens): start a new conversation")
        self.emit("brain_start", step=step, model=vllm_label(brain, model_id), think=think, prompt_tokens=len(ids))
        req = {"model": model_id, "prompt": ids, "stream": True, "max_tokens": min(2048, budget),
               # not greedy: small models distilled from R1 get stuck in loops
               "temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1,
               "skip_special_tokens": False, "stop_token_ids": [2, 4]}
        full, sent_think, sent_content, tokens, t0 = "", 0, 0, 0, time.monotonic()
        with self.http.stream("POST", f"{base}/v1/completions", json=req) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if self.cancelled:
                    break
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                full += json.loads(line[6:])["choices"][0]["text"]
                tokens += 1
                reasoning, content = split_think(SPECIAL_RE.sub("", full), think)
                # until </think> arrives, the tail may be its beginning, so we hold it back for a moment
                upto = len(reasoning) if content is not None else len(reasoning) - len("</think>")
                if upto > sent_think:
                    self.emit("brain_token", step=step, kind="think", text=reasoning[sent_think:upto])
                    sent_think = upto
                if content and len(content) > sent_content:
                    self.emit("brain_token", step=step, kind="content", text=content[sent_content:])
                    sent_content = len(content)
        reasoning, content = split_think(SPECIAL_RE.sub("", full), think)
        if content is None:  # the token limit ran out while thinking
            if len(reasoning) > sent_think:
                self.emit("brain_token", step=step, kind="think", text=reasoning[sent_think:])
            content = ""
        seconds = time.monotonic() - t0
        self.emit("brain_end", step=step, tokens=tokens, seconds=round(seconds, 1),
                  tok_s=round(tokens / seconds) if seconds else 0)
        return content.strip()

    def _generate_ollama(self, brain, think, step):
        """Model in Ollama with native tools: Ollama itself parses the calls and separates the thinking."""
        can_think = "thinking" in capabilities(brain["base"], brain["model"])
        body = {"model": brain["model"], "messages": to_ollama(self.messages), "tools": TOOLS, "stream": True,
                "options": {"num_ctx": 8192}}  # sampling stays at the model's Modelfile defaults
        if can_think:
            body["think"] = think
        self.emit("brain_start", step=step, model=brain["label"], think=think and can_think)
        content, calls, tokens, t0 = "", [], 0, time.monotonic()
        with self.http.stream("POST", f"{brain['base']}/api/chat", json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"{brain['label']}: {r.text}")
            for line in r.iter_lines():
                if self.cancelled:
                    break
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"{brain['label']}: {chunk['error']}")
                msg = chunk.get("message", {})
                if msg.get("thinking"):
                    self.emit("brain_token", step=step, kind="think", text=msg["thinking"])
                if msg.get("content"):
                    content += msg["content"]
                    self.emit("brain_token", step=step, kind="content", text=msg["content"])
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    calls.append((f"call_{step}_{len(calls)}", str(fn.get("name", "")),
                                  decode_args(fn.get("arguments", {}))))
                if chunk.get("done"):
                    tokens = chunk.get("eval_count", 0)
        seconds = time.monotonic() - t0
        self.emit("brain_end", step=step, tokens=tokens, seconds=round(seconds, 1),
                  tok_s=round(tokens / seconds) if seconds else 0)
        return content.strip(), calls
