#!/usr/bin/env python3
"""Field Notes - the laptop side of the pocket voice recorder.

Opens a local app window. From there you can:
  * import notes from the Pico recorder (or add voice notes from your phone),
  * listen to them,
  * transcribe them with Whisper,
  * have Gemma (through Ollama) write the journal entry for each outing and a weekly recap.

Everything runs on this laptop. Nothing is sent anywhere.

    python app.py            # opens the app
    FIELDNOTES_PORT=9000 python app.py

Needs: mpremote (already in your venv). For transcription: `pip install faster-whisper`
(or whisper.cpp's whisper-cli). For the journal: Ollama with a Gemma model.
"""

import glob
import importlib.util
import json
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import wave
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FIELDNOTES_DATA", HERE / "data"))
WALKS = DATA / "walks"
INBOX = DATA / "inbox"
CONFIG_PATH = DATA / "config.json"
WEEK_PATH = DATA / "week.json"
WEATHER_PATH = DATA / "weather.json"
PORT = int(os.environ.get("FIELDNOTES_PORT", "8765"))
OUT_RATE = 16000                      # what Whisper wants

DEFAULTS = {
    "whisper_model": "base.en",       # faster-whisper model name, or a ggml .bin path for whisper.cpp
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "",               # empty = pick the best Gemma that's installed
    "clear_after_import": True,
    "auto_process": True,             # transcribe + write the entry right after an import
    "city": "Pune",                   # weather banner; needs SERPAPI_KEY (environment or .env)
    "pincode": "",                    # optional; picks "city" from the areas of an Indian PIN code
    "max_temp_c": 34,                 # comfort limits Gemma weighs for the weather verdict
    "max_precip_pct": 40,
}

try:
    import numpy as np
except ImportError:                   # the app still works; whine removal is skipped
    np = None


# ======================================================================================
# config and storage
# ======================================================================================

def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    DATA.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


_store_lock = threading.Lock()


def walk_dir(walk_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", walk_id or ""):
        raise ValueError("bad walk id")
    return WALKS / walk_id


def load_walk(walk_id):
    return json.loads((walk_dir(walk_id) / "walk.json").read_text())


def save_walk(walk):
    with _store_lock:
        d = walk_dir(walk["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "walk.json.tmp"
        tmp.write_text(json.dumps(walk, indent=2))
        tmp.replace(d / "walk.json")


def all_walks():
    walks = []
    for p in WALKS.glob("*/walk.json"):
        try:
            walks.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    walks.sort(key=lambda w: w.get("date", ""), reverse=True)
    return walks


def walk_stats(walk):
    notes = walk.get("notes", [])
    if not notes:
        return {"notes": 0, "spoken_s": 0, "span_min": 0}
    first = min(n["start_s"] for n in notes)
    last = max(n["start_s"] + n["duration"] for n in notes)
    return {
        "notes": len(notes),
        "spoken_s": round(sum(n["duration"] for n in notes)),
        "span_min": max(1, round((last - first) / 60)),
    }


# ======================================================================================
# audio: read the recorder's mu-law WAVs (or anything ffmpeg reads), clean up, write 16 kHz
# ======================================================================================

def _ulaw_to_lin(byte):
    u = ~byte & 0xFF
    sign, exp, mant = u & 0x80, (u >> 4) & 7, u & 0x0F
    s = (((mant << 3) + 0x84) << exp) - 0x84
    return -s if sign else s


_ULAW = [_ulaw_to_lin(b) for b in range(256)]


def read_wav(path):
    """Return (rate, list of int samples, 16-bit scale) for PCM 8/16-bit or mu-law WAV. None if unsupported."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if not fmt or payload is None:
        return None
    tag, channels, rate, _, _, bits = fmt
    if tag == 7 and bits == 8:
        samples = [_ULAW[b] for b in payload]
    elif tag == 1 and bits == 16:
        samples = list(struct.unpack("<%dh" % (len(payload) // 2), payload[: len(payload) // 2 * 2]))
    elif tag == 1 and bits == 8:
        samples = [(b - 128) << 8 for b in payload]
    else:
        return None
    if channels > 1:
        samples = samples[::channels]
    return rate, samples


def ffmpeg_to_pcm(path):
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("This file type needs ffmpeg (sudo pacman -S ffmpeg), or record as WAV.")
    out = subprocess.run([ff, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(OUT_RATE),
                          "-f", "s16le", "-"], capture_output=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError("ffmpeg could not read %s: %s" % (Path(path).name, out.stderr.decode()[-200:]))
    raw = out.stdout
    return OUT_RATE, list(struct.unpack("<%dh" % (len(raw) // 2), raw[: len(raw) // 2 * 2]))


def remove_whine(x, rate):
    """Notch out steady tones (the electrical whine). Needs numpy; speech varies so it survives."""
    n = len(x)
    if np is None or n < rate:
        return x
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    hz_per_bin = rate / n
    w = max(3, int(150 / hz_per_bin))
    c = np.cumsum(np.concatenate([[0.0], mag]))
    lo = np.clip(np.arange(len(mag)) - w // 2, 0, len(mag))
    hi = np.clip(np.arange(len(mag)) + w // 2 + 1, 0, len(mag))
    smooth = (c[hi] - c[lo]) / (hi - lo)
    tonal = (mag > 25 * smooth) & (freqs > 300)
    half = max(1, int(4 / hz_per_bin))
    for i in np.flatnonzero(tonal)[:400]:
        spec[max(0, i - half): i + half + 1] = 0
    if rate <= 8000:
        spec[freqs > 3700] = 0             # top of the recorder's band, where most whine lives
    return np.fft.irfft(spec, n)


def clean_audio(rate, samples, denoise=True):
    """DC removal, whine removal, gentle normalise, resample to 16 kHz. Returns int16 list."""
    if np is not None:
        x = np.asarray(samples, dtype=np.float64)
        x -= x.mean()
        # one-pole high-pass at ~80 Hz removes rumble and the slow drift of the bias
        a = np.exp(-2 * np.pi * 80 / rate)
        y = np.empty_like(x)
        prev_x = prev_y = 0.0
        for i, v in enumerate(x):
            prev_y = a * (prev_y + v - prev_x)
            prev_x = v
            y[i] = prev_y
        if denoise:
            y = remove_whine(y, rate)
        if rate != OUT_RATE:
            t_out = np.arange(int(len(y) * OUT_RATE / rate)) * (rate / OUT_RATE)
            y = np.interp(t_out, np.arange(len(y)), y)
        peak = np.max(np.abs(y)) if len(y) else 0
        if peak > 0:
            y *= min(20.0, 30000 / peak)
        return np.clip(y, -32768, 32767).astype(np.int16).tolist()
    # pure-Python fallback
    mean = sum(samples) / max(1, len(samples))
    x = [s - mean for s in samples]
    if rate != OUT_RATE:
        ratio = rate / OUT_RATE
        n_out = int(len(x) / ratio)
        x = [x[min(len(x) - 1, int(i * ratio))] for i in range(n_out)]
    peak = max((abs(v) for v in x), default=0)
    g = min(20.0, 30000 / peak) if peak else 1.0
    return [max(-32768, min(32767, int(v * g))) for v in x]


def write_wav16(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))


def convert_note(src, dst, denoise=True):
    """Any supported audio -> cleaned 16 kHz WAV. Returns duration in seconds."""
    got = read_wav(src) if str(src).lower().endswith(".wav") else None
    rate, samples = got if got else ffmpeg_to_pcm(src)
    out = clean_audio(rate, samples, denoise)
    write_wav16(dst, out)
    return round(len(out) / OUT_RATE, 1)


# ======================================================================================
# the recorder (Pico over USB, via mpremote)
# ======================================================================================

DEVICE_INFO = (
    "import os, json\n"
    "try:\n files = os.listdir('clips')\nexcept OSError:\n files = []\n"
    "idx = ''\n"
    "try:\n idx = open('clips/index.csv').read()\nexcept OSError:\n pass\n"
    "st = os.statvfs('/')\n"
    "print('@@' + json.dumps({'free': st[0]*st[3], 'total': st[0]*st[2], 'files': files, 'index': idx}))\n"
)
DEVICE_CLEAR = "import os\nfor f in os.listdir('clips'):\n os.remove('clips/' + f)\nprint('@@cleared')\n"


def find_recorder():
    """The Pico's serial port, found by its USB vendor id (2e8a). None if not plugged in."""
    for dev in sorted(glob.glob("/dev/ttyACM*")):
        p = os.path.realpath("/sys/class/tty/%s/device" % os.path.basename(dev))
        for _ in range(5):
            vid = os.path.join(p, "idVendor")
            if os.path.exists(vid):
                try:
                    if open(vid).read().strip().lower() == "2e8a":
                        return dev
                except OSError:
                    pass
                break
            p = os.path.dirname(p)
    return None


def mpremote(port, *args, timeout=120):
    if importlib.util.find_spec("mpremote"):
        cmd = [sys.executable, "-m", "mpremote"]
    elif shutil.which("mpremote"):
        cmd = [shutil.which("mpremote")]
    else:
        raise RuntimeError("mpremote isn't installed in this Python (pip install mpremote).")
    out = subprocess.run(cmd + ["connect", port, *args], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        msg = (out.stderr or out.stdout).strip().splitlines()
        raise RuntimeError("The recorder didn't answer (%s). Reseat the USB plug and try again."
                           % (msg[-1] if msg else "no output"))
    return out.stdout


def _parse_index(text):
    rows = {}
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[0].endswith(".wav"):
            try:
                rows[parts[0]] = (int(parts[1]), int(parts[2]), float(parts[3]))
            except ValueError:
                pass
    return rows


# ======================================================================================
# transcription: faster-whisper if installed, else whisper.cpp's whisper-cli
# ======================================================================================

class Transcriber:
    def __init__(self):
        self._model = None
        self._model_name = None

    def status(self, cfg):
        if importlib.util.find_spec("faster_whisper"):
            return {"ok": True, "kind": "faster-whisper", "detail": cfg["whisper_model"]}
        cli = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
        model = self._cpp_model(cfg)
        if cli and model:
            return {"ok": True, "kind": "whisper.cpp", "detail": Path(model).name}
        return {"ok": False, "kind": None,
                "detail": "Not installed. In your venv: pip install faster-whisper"}

    @staticmethod
    def _cpp_model(cfg):
        m = cfg["whisper_model"]
        if m.endswith(".bin") and os.path.exists(os.path.expanduser(m)):
            return os.path.expanduser(m)
        for pat in ("~/whisper.cpp/models/ggml-*.bin", "~/.local/share/whisper.cpp/ggml-*.bin",
                    "/usr/share/whisper.cpp*/ggml-*.bin", "/usr/share/whisper.cpp*/*/ggml-*.bin"):
            found = sorted(glob.glob(os.path.expanduser(pat)))
            if found:
                return found[0]
        return None

    def transcribe(self, wav_path, cfg):
        st = self.status(cfg)
        if not st["ok"]:
            raise RuntimeError("Whisper isn't installed. In your venv: pip install faster-whisper")
        lang = "en" if cfg["whisper_model"].endswith(".en") else None
        if st["kind"] == "faster-whisper":
            from faster_whisper import WhisperModel
            if self._model is None or self._model_name != cfg["whisper_model"]:
                self._model = WhisperModel(cfg["whisper_model"], device="cpu", compute_type="int8")
                self._model_name = cfg["whisper_model"]
            # hand Whisper the samples ourselves: newer PyAV breaks faster-whisper's own decoder
            with wave.open(str(wav_path), "rb") as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            segments, _ = self._model.transcribe(pcm.astype(np.float32) / 32768.0, beam_size=1,
                                                 language=lang, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()
        cli = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
        out = subprocess.run([cli, "-m", self._cpp_model(cfg), "-f", str(wav_path), "-nt", "-np",
                              "-l", "en" if lang else "auto"], capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            raise RuntimeError("whisper.cpp failed: " + out.stderr.strip()[-200:])
        return " ".join(line.strip() for line in out.stdout.splitlines() if line.strip())


TRANSCRIBER = Transcriber()


# ======================================================================================
# Gemma through Ollama
# ======================================================================================

def ollama_models(cfg):
    try:
        with urllib.request.urlopen(cfg["ollama_url"] + "/api/tags", timeout=2) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except (OSError, ValueError):
        return None


def pick_model(cfg, models):
    if not models:
        return None
    if cfg["ollama_model"] in models:
        return cfg["ollama_model"]
    for pref in ("gemma4:e4b", "gemma4:e2b", "gemma4", "gemma3:4b", "gemma3", "gemma"):
        for m in models:
            if m.startswith(pref):
                return m
    return models[0]


def gemma_status(cfg):
    models = ollama_models(cfg)
    if models is None:
        return {"ok": False, "model": None, "models": [],
                "detail": "Ollama isn't running. Start it with: ollama serve"}
    model = pick_model(cfg, models)
    if not model:
        return {"ok": False, "model": None, "models": [],
                "detail": "No model yet. Run: ollama pull gemma3:4b"}
    return {"ok": True, "model": model, "models": models, "detail": model}


def ask_gemma(cfg, prompt, schema, temperature=0.5):
    st = gemma_status(cfg)
    if not st["ok"]:
        raise RuntimeError(st["detail"])
    body = {"model": st["model"], "prompt": prompt, "stream": False, "format": schema,
            "options": {"temperature": temperature, "num_ctx": 4096}}
    req = urllib.request.Request(cfg["ollama_url"] + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r).get("response", "")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        if isinstance(e, urllib.error.HTTPError) and e.code == 400:
            text = None
        else:
            raise RuntimeError("Gemma stopped answering (%s). Is ollama serve still running?" % e)
    if text is None:
        body["format"] = "json"                    # older Ollama: no schema support, ask for plain JSON
        req = urllib.request.Request(req.full_url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r).get("response", "")
    try:
        return json.loads(text), st["model"]
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0)), st["model"]
        raise RuntimeError("Gemma's reply wasn't readable. Try again.")


ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "entry": {"type": "string"},
        "next_time": {"type": "string"},
        "goals": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string"},
    },
    "required": ["title", "entry", "next_time", "goals", "mood"],
}

WEEK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "recap": {"type": "string"},
        "highlight": {"type": "string"},
        "goal_check": {"type": "array", "items": {
            "type": "object",
            "properties": {"goal": {"type": "string"}, "status": {"type": "string"}},
            "required": ["goal", "status"]}},
        "next_week": {"type": "string"},
    },
    "required": ["title", "recap", "highlight", "goal_check", "next_week"],
}


def fmt_offset(seconds):
    m, s = divmod(int(seconds), 60)
    return "%d:%02d" % (m, s)


def entry_prompt(walk):
    lines = ["[%s in] %s" % (fmt_offset(n["start_s"]), n["transcript"].strip())
             for n in walk["notes"] if (n.get("transcript") or "").strip()]
    return (
        "You keep a private outdoor journal for one person. While they were outside, with their "
        "phone put away, they spoke short voice notes into a pocket recorder. Here are the "
        "transcripts in order; the time in brackets is how long after setting out they spoke.\n\n"
        + "\n".join(lines) +
        "\n\nWrite their journal entry for this outing.\n"
        "Rules:\n"
        "- First person, past tense, plain and warm, like someone writing in their own notebook. "
        "60 to 140 words.\n"
        "- Use only what the notes say. Do not invent places, people, food, weather or feelings.\n"
        "- Transcripts may contain small errors; read them sensibly.\n"
        "- title: 2 to 6 words.\n"
        "- next_time: one small, concrete thing to try or look for on the next outing, taken from "
        "something they mentioned. One sentence, addressed to them as 'you'.\n"
        "- goals: goals they stated out loud (for example 'goal: four walks this week'). "
        "Empty list if none.\n"
        "- mood: one or two words for how they sounded.\n"
        "Reply as JSON with keys title, entry, next_time, goals, mood."
    )


def week_prompt(walks):
    parts = []
    goals = []
    for w in sorted(walks, key=lambda w: w["date"]):
        j = w["journal"]
        day = datetime.fromisoformat(w["date"]).strftime("%A")
        parts.append("%s - %s: %s" % (day, j["title"], j["entry"]))
        goals += [g for g in j.get("goals", []) if g.lower() not in {x.lower() for x in goals}]
    return (
        "Here are one person's outdoor journal entries from the past week, written from voice "
        "notes they spoke while outside with their phone put away.\n\n" + "\n\n".join(parts) +
        "\n\nGoals they set: " + ("; ".join(goals) if goals else "none") +
        "\n\nWrite a short weekly recap, in second person ('you').\n"
        "- title: 2 to 6 words.\n"
        "- recap: 80 to 150 words, warm and plain, only from what the entries say.\n"
        "- highlight: the single best moment of the week, one sentence.\n"
        "- goal_check: for each goal, a short status based on the entries (met, partly, not yet).\n"
        "- next_week: one gentle suggestion for next week, one sentence.\n"
        "Reply as JSON with keys title, recap, highlight, goal_check, next_week."
    )


# ======================================================================================
# background jobs (one at a time, so the laptop stays usable)
# ======================================================================================

JOBS = queue.Queue()
JOB = {"busy": False, "label": "", "message": "", "error": "", "done": "", "queued": 0}
_job_lock = threading.Lock()


def job_update(**kw):
    with _job_lock:
        JOB.update(kw)


def enqueue(label, fn, *args):
    with _job_lock:
        JOB["queued"] += 1
    JOBS.put((label, fn, args))


def worker():
    while True:
        label, fn, args = JOBS.get()
        with _job_lock:
            JOB["queued"] -= 1
        job_update(busy=True, label=label, message=label + "...", error="")
        try:
            done = fn(*args)
            job_update(done=done or (label + ": done."))
        except Exception as e:
            traceback.print_exc()
            job_update(error="%s: %s" % (type(e).__name__, e))
        finally:
            job_update(busy=False, message="")


def new_walk_id(suffix):
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + suffix


def job_import_recorder(clear):
    cfg = load_config()
    port = find_recorder()
    if not port:
        raise RuntimeError("The recorder isn't plugged in.")
    job_update(message="Asking the recorder what it has...")
    out = mpremote(port, "exec", DEVICE_INFO)
    info = json.loads(next(l[2:] for l in out.splitlines() if l.startswith("@@")))
    wavs = sorted(f for f in info["files"] if f.endswith(".wav"))
    if not wavs:
        mpremote(port, "reset")
        return "No new notes on the recorder."

    tmp = Path(tempfile.mkdtemp(prefix="fieldnotes-"))
    try:
        job_update(message="Copying %d notes off the recorder..." % len(wavs))
        mpremote(port, "cp", "-r", ":clips", str(tmp) + "/", timeout=600)
        wid = import_recorder_clips(tmp / "clips", wavs, info["index"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if clear:
        job_update(message="Clearing the recorder...")
        mpremote(port, "exec", DEVICE_CLEAR)
    job_update(message="Restarting the recorder...")
    try:
        mpremote(port, "reset", timeout=20)
    except Exception:
        pass                                     # it restarts by itself on the next replug anyway
    if cfg["auto_process"]:
        queue_processing(wid)
    return "Imported %d notes as one outing." % len(load_walk(wid)["notes"])


def import_recorder_clips(src_dir, wavs, index_text):
    """Recorder clips + their index.csv -> one outing (one import = one outing). Returns its id."""
    index = _parse_index(index_text)
    clips, last = [], None
    for f in sorted(wavs):                       # clip numbers count up on the recorder
        got = read_wav(src_dir / f)
        length = len(got[1]) / got[0] if got else 0
        if f in index:
            boot, t_boot, _ = index[f]
            guessed = False
        elif last:                               # no index line: assume it came right after the previous clip
            boot, t_boot, guessed = last["boot"], last["t_boot"] + round(last["length"]), True
        else:
            boot, t_boot, guessed = 0, 0, True
        last = {"name": f, "boot": boot, "t_boot": t_boot, "length": length, "guessed": guessed}
        clips.append(last)
    clips.sort(key=lambda c: (c["boot"], c["t_boot"]))

    wid = new_walk_id("b%d" % clips[0]["boot"])
    d = walk_dir(wid)
    (d / "raw").mkdir(parents=True, exist_ok=True)
    if index_text:
        (d / "raw" / "index.csv").write_text(index_text)
    notes = []
    for k, c in enumerate(clips, 1):
        job_update(message="Cleaning up audio: note %d of %d" % (k, len(clips)))
        shutil.copy2(src_dir / c["name"], d / "raw" / c["name"])
        out_name = "note_%02d.wav" % k
        dur = convert_note(src_dir / c["name"], d / out_name)
        note = {"file": out_name, "raw": "raw/" + c["name"], "boot": c["boot"], "t_boot": c["t_boot"],
                "start_s": c["t_boot"], "duration": dur, "transcript": None}
        if c["guessed"]:
            note["guessed"] = True
        notes.append(note)
    walk = {"id": wid, "date": datetime.now().isoformat(timespec="seconds"), "source": "recorder",
            "boot": clips[0]["boot"], "notes": notes, "journal": None}
    retime(walk)
    save_walk(walk)
    return wid


def retime(walk):
    """Put every note on one timeline, ordered by (boot, seconds since power-on).
    The recorder's clock restarts at 0 on every boot, so each later boot is placed after the
    previous boot's last note; times after a restart (or with no index line) are estimates.
    Notes with no boot (phone notes) keep their own spacing, after the recorder's."""
    notes = walk["notes"]
    for n in notes:
        n.setdefault("boot", walk.get("boot"))
        n.setdefault("t_boot", n["start_s"])
    notes.sort(key=lambda n: (n["boot"] is None, n["boot"] or 0, n["t_boot"]))
    segment, offset, seg_t0, prev_end = -1, 0, 0, 0
    for i, n in enumerate(notes):
        if i == 0 or n["boot"] != notes[i - 1]["boot"]:
            segment += 1
            offset = prev_end if segment else 0
            seg_t0 = n["t_boot"] if n["boot"] is None else 0
        n["start_s"] = int(round(offset + n["t_boot"] - seg_t0))
        n["estimated"] = segment > 0 or bool(n.get("guessed"))
        prev_end = max(prev_end, n["start_s"] + n["duration"])
    walk["restarts"] = max(0, len({n["boot"] for n in notes if n["boot"] is not None}) - 1)
    return walk


def job_process_upload(batch):
    cfg = load_config()
    src = INBOX / batch
    meta = json.loads((src / "meta.json").read_text())
    index = [m for m in meta if m["name"].lower() == "index.csv"]
    if index:                                    # a saved recorder folder: import it like the recorder
        clips = src / "clips"
        clips.mkdir(exist_ok=True)
        for m in meta:
            (src / m["stored"]).replace(clips / m["name"])
        wavs = sorted(m["name"] for m in meta if m["name"].lower().endswith(".wav"))
        if not wavs:
            raise RuntimeError("That folder has an index.csv but no recordings.")
        wid = import_recorder_clips(clips, wavs, (clips / index[0]["name"]).read_text())
        shutil.rmtree(src, ignore_errors=True)
        if cfg["auto_process"]:
            queue_processing(wid)
        return "Imported %d notes from the saved recorder folder as one outing." % len(wavs)
    files = sorted(meta, key=lambda m: m["mtime"])
    if not files:
        return "Nothing to add."
    t0 = files[0]["mtime"] / 1000
    wid = new_walk_id("phone")
    d = walk_dir(wid)
    (d / "raw").mkdir(parents=True, exist_ok=True)
    notes = []
    for k, m in enumerate(files, 1):
        job_update(message="Cleaning up audio: note %d of %d" % (k, len(files)))
        raw_name = "%02d_%s" % (k, m["name"])
        shutil.copy2(src / m["stored"], d / "raw" / raw_name)
        out_name = "note_%02d.wav" % k
        dur = convert_note(src / m["stored"], d / out_name, denoise=False)
        notes.append({"file": out_name, "raw": "raw/" + raw_name,
                      "start_s": int(m["mtime"] / 1000 - t0), "duration": dur, "transcript": None})
    save_walk({"id": wid, "date": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
               "source": "phone", "boot": None, "notes": notes, "journal": None})
    shutil.rmtree(src, ignore_errors=True)
    if cfg["auto_process"]:
        queue_processing(wid)
    return "Added %d phone notes." % len(notes)


def job_transcribe(walk_id):
    cfg = load_config()
    walk = load_walk(walk_id)
    pending = [n for n in walk["notes"] if n.get("transcript") is None]
    for k, n in enumerate(pending, 1):
        job_update(message="Transcribing note %d of %d%s" % (
            k, len(pending), " (first run downloads the Whisper model)" if k == 1 else ""))
        text = TRANSCRIBER.transcribe(walk_dir(walk_id) / n["file"], cfg)
        walk = load_walk(walk_id)                       # re-read: the user may have edited meanwhile
        for m in walk["notes"]:
            if m["file"] == n["file"]:
                m["transcript"] = text
        save_walk(walk)
    return "Transcribed %d notes." % len(pending)


def job_entry(walk_id):
    cfg = load_config()
    walk = load_walk(walk_id)
    if not any((n.get("transcript") or "").strip() for n in walk["notes"]):
        raise RuntimeError("Nothing to write from yet: the notes have no transcripts.")
    job_update(message="Gemma is writing the journal entry...")
    reply, model = ask_gemma(cfg, entry_prompt(walk), ENTRY_SCHEMA)
    walk = load_walk(walk_id)
    walk["journal"] = {
        "title": str(reply.get("title", "")).strip() or "An outing",
        "entry": str(reply.get("entry", "")).strip(),
        "next_time": str(reply.get("next_time", "")).strip(),
        "goals": [str(g).strip() for g in reply.get("goals", []) if str(g).strip()],
        "mood": str(reply.get("mood", "")).strip(),
        "model": model,
        "written": datetime.now().isoformat(timespec="seconds"),
    }
    save_walk(walk)
    write_markdown(walk)
    return "Journal entry written."


def job_week():
    cfg = load_config()
    since = datetime.now() - timedelta(days=7)
    walks = [w for w in all_walks() if w.get("journal") and datetime.fromisoformat(w["date"]) >= since]
    if not walks:
        raise RuntimeError("No journal entries from the past 7 days yet.")
    job_update(message="Gemma is writing the weekly recap...")
    reply, model = ask_gemma(cfg, week_prompt(walks), WEEK_SCHEMA)
    stats = [walk_stats(w) for w in walks]
    week = {
        "title": str(reply.get("title", "")).strip() or "Your week outside",
        "recap": str(reply.get("recap", "")).strip(),
        "highlight": str(reply.get("highlight", "")).strip(),
        "goal_check": [g for g in reply.get("goal_check", []) if isinstance(g, dict)],
        "next_week": str(reply.get("next_week", "")).strip(),
        "outings": len(walks),
        "notes": sum(s["notes"] for s in stats),
        "minutes_outside": sum(s["span_min"] for s in stats),
        "model": model,
        "written": datetime.now().isoformat(timespec="seconds"),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    WEEK_PATH.write_text(json.dumps(week, indent=2))
    return "Weekly recap written."


def job_process(walk_id):
    """Transcribe, then write the entry; stop with the real error if transcription fails."""
    try:
        done = job_transcribe(walk_id)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Transcription failed: %s: %s" % (type(e).__name__, e)) from e
    if not gemma_status(load_config())["ok"]:
        return done + " Gemma isn't available, so no journal entry yet."
    return job_entry(walk_id)


def queue_processing(walk_id):
    if TRANSCRIBER.status(load_config())["ok"]:
        enqueue("Transcribing and writing the journal entry", job_process, walk_id)


def merge_walks(into_id, from_id):
    """Move one outing's notes into another, retime them, and write the entry afresh."""
    if into_id == from_id:
        raise ValueError("Pick two different outings.")
    with _job_lock:
        if JOB["busy"] or JOB["queued"]:
            raise ValueError("Wait for the current job to finish, then merge.")
    into, src = load_walk(into_id), load_walk(from_id)
    d_into, d_src = walk_dir(into_id), walk_dir(from_id)
    for w in (into, src):
        for n in w["notes"]:
            n.setdefault("boot", w.get("boot"))
            if "t_boot" not in n and n["boot"] is None:      # phone notes: real clock time
                n["t_boot"] = datetime.fromisoformat(w["date"]).timestamp() + n["start_s"]
    (d_into / "raw").mkdir(exist_ok=True)
    taken = {n["file"] for n in into["notes"]}
    k = 1
    for n in src["notes"]:
        while "note_%02d.wav" % k in taken:
            k += 1
        new = "note_%02d.wav" % k
        taken.add(new)
        shutil.move(d_src / n["file"], d_into / new)
        n["file"] = new
        if n.get("raw") and (d_src / n["raw"]).exists():
            dst = d_into / "raw" / Path(n["raw"]).name
            if dst.exists():
                dst = d_into / "raw" / ("%s_%s" % (from_id, dst.name))
            shutil.move(d_src / n["raw"], dst)
            n["raw"] = "raw/" + dst.name
        into["notes"].append(n)
    for f in sorted((d_src / "raw").glob("*")):               # e.g. the copied index.csv
        dst = d_into / "raw" / f.name
        shutil.move(f, dst if not dst.exists() else d_into / "raw" / ("%s_%s" % (from_id, f.name)))
    into["date"] = min(into["date"], src["date"])
    into["journal"] = None
    retime(into)
    save_walk(into)
    (d_into / "journal.md").unlink(missing_ok=True)
    (d_src / "walk.json").unlink()                             # only metadata is left there now
    (d_src / "journal.md").unlink(missing_ok=True)
    for p in (d_src / "raw", d_src):
        try:
            p.rmdir()
        except OSError:
            pass
    queue_processing(into_id)


# ======================================================================================
# "Go touch grass": the weather from SerpApi (only the city is sent), the verdict from Gemma
# ======================================================================================

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "go": {"type": "string", "enum": ["now", "later", "stay"]},
        "time": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["go", "time", "reason"],
}
_weather = {"busy": False, "tried": None, "error": ""}
_weather_lock = threading.Lock()


def load_env_file(path=HERE / ".env"):
    """KEY=value lines from fieldnotes/.env (kept out of git); real environment variables win."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def serpapi_key():
    return os.environ.get("SERPAPI_KEY", "").strip()


def _num(text):
    m = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    return float(m.group(0)) if m else None


_pin_cache = {}


def pincode_areas(pin):
    """Areas (post offices) for an Indian PIN code, from India Post's public API. Only the PIN is sent."""
    if not re.fullmatch(r"[1-9]\d{5}", pin or ""):
        raise ValueError("A PIN code is 6 digits.")
    if pin not in _pin_cache:
        with urllib.request.urlopen("https://api.postalpincode.in/pincode/" + pin, timeout=15) as r:
            reply = json.load(r)
        offices = (reply[0].get("PostOffice") if reply else None) or []
        _pin_cache[pin] = [{"name": o.get("Name", ""), "district": o.get("District", ""),
                            "state": o.get("State", "")} for o in offices if o.get("Name")]
    return _pin_cache[pin]


def fetch_weather(cfg):
    q = urlencode({"engine": "google", "q": "weather " + cfg["city"], "api_key": serpapi_key()})
    with urllib.request.urlopen("https://serpapi.com/search.json?" + q, timeout=30) as r:
        reply = json.load(r)
    ab = reply.get("answer_box") or {}
    if ab.get("type") != "weather_result":
        raise RuntimeError(reply.get("error") or "No weather for %s in the search results." % cfg["city"])
    fahrenheit = str(ab.get("unit", "")).lower().startswith("f")

    def hour(h):
        t = _num(h.get("temperature"))
        if t is not None and fahrenheit:
            t = (t - 32) * 5 / 9
        pct = lambda k: None if _num(h.get(k)) is None else round(_num(h.get(k)))
        return {"time": " ".join(str(h.get("time", "")).split()),   # Google uses odd spaces ("9:00\u202fPM")
                "temp_c": None if t is None else round(t), "weather": h.get("weather", ""),
                "precip_pct": pct("precipitation"), "humidity_pct": pct("humidity"), "wind": h.get("wind", "")}

    now = hour(dict(ab, time=ab.get("date", "")))
    later = [h for h in map(hour, ab.get("hourly_forecast") or []) if h["time"] != now["time"]][:3]
    return {"city": cfg["city"], "location": ab.get("location", cfg["city"]), "now": now,
            "next_hours": later, "fetched": time.time()}


def describe_hour(h):
    return "%s: %s°C, %s, %s%% chance of rain, humidity %s%%, wind %s" % (
        h["time"], h["temp_c"], h["weather"] or "?", h["precip_pct"], h["humidity_pct"], h["wind"])


def weather_prompt(w, cfg):
    return (
        "Someone is deciding whether to go outside for a walk, with their phone put away. "
        "Here is the weather in %s.\n\nNow (%s)\n\nNext hours:\n%s\n\n"
        "Their comfort limits: no hotter than %s°C, chance of rain at most %s%%.\n"
        "Rule: if the weather right now is within both limits, the answer is go now.\n"
        "Otherwise pick the first of the next hours that is within the limits (go later), "
        "or stay in if none is.\n"
        "- go: \"now\", \"later\" or \"stay\".\n"
        "- time: if later, the hour to go, copied from the list (e.g. 'Monday 7:00 AM'); else empty.\n"
        "- reason: a few plain words, e.g. 'rain until 6' or 'clear and mild'.\n"
        "Reply as JSON with keys go, time, reason."
        % (w["location"], describe_hour(w["now"]),
           "\n".join(describe_hour(h) for h in w["next_hours"]) or "(no hourly forecast)",
           cfg["max_temp_c"], cfg["max_precip_pct"]))


def refresh_weather(cfg):
    try:
        w = fetch_weather(cfg)
        try:
            reply, model = ask_gemma(cfg, weather_prompt(w, cfg), VERDICT_SCHEMA, temperature=0.1)
            go = str(reply.get("go", "")).strip().lower()
            when = re.sub(r"^[A-Za-z]+day\s+", "", str(reply.get("time", "")).strip())
            reason = str(reply.get("reason", "")).strip()
            w["go"] = go if go in ("now", "later", "stay") else None
            w["verdict"] = ("Time to touch grass!!" if go == "now"
                            else ("Go later, around %s" % when if when else "Go later") if go == "later"
                            else "Stay in (%s)" % reason if go == "stay"
                            else None)
            w["reason"], w["model"] = reason, model
        except Exception as e:
            traceback.print_exc()
            w["verdict"], w["reason"] = None, "Gemma couldn't decide (%s)." % e
        DATA.mkdir(parents=True, exist_ok=True)
        WEATHER_PATH.write_text(json.dumps(w, indent=2))
        _weather["error"] = ""
    except Exception as e:                              # keep the last cached value
        traceback.print_exc()
        _weather["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        _weather["busy"] = False


def weather_state(cfg):
    """What the banner shows. Starts a background refresh at most once an hour (or on a new city/key)."""
    if not serpapi_key() or not str(cfg.get("city", "")).strip():
        return None
    try:
        w = json.loads(WEATHER_PATH.read_text())
    except (OSError, ValueError):
        w = None
    attempt = (cfg["city"], serpapi_key())
    fresh = w and w.get("city") == cfg["city"] and time.time() - w.get("fetched", 0) < 3600
    with _weather_lock:
        tried = _weather["tried"]
        recent = tried and tried[0] == attempt and time.time() - tried[1] < 3600
        if not fresh and not recent and not _weather["busy"]:
            _weather.update(busy=True, tried=(attempt, time.time()))
            threading.Thread(target=refresh_weather, args=(dict(cfg),), daemon=True).start()
    if w and w.get("city") != cfg["city"]:
        w = None
    return {"data": w, "busy": _weather["busy"], "error": _weather["error"]}


def write_markdown(walk):
    j = walk["journal"]
    date = datetime.fromisoformat(walk["date"]).strftime("%A %d %B %Y")
    lines = ["# %s" % j["title"], "", "*%s*" % date, "", j["entry"], ""]
    if j["next_time"]:
        lines += ["**Next time:** " + j["next_time"], ""]
    if j["goals"]:
        lines += ["**Goals:** " + "; ".join(j["goals"]), ""]
    lines += ["---", ""]
    for n in walk["notes"]:
        lines.append("- `%s in` %s" % (fmt_offset(n["start_s"]), n.get("transcript") or ""))
    (walk_dir(walk["id"]) / "journal.md").write_text("\n".join(lines) + "\n")


# ======================================================================================
# web server
# ======================================================================================

def state_payload():
    cfg = load_config()
    port = find_recorder()
    walks = all_walks()
    for w in walks:
        w["stats"] = walk_stats(w)
    week = None
    try:
        week = json.loads(WEEK_PATH.read_text())
    except (OSError, ValueError):
        pass
    with _job_lock:
        job = dict(JOB)
    return {
        "recorder": {"port": port,
                     "access": bool(port and os.access(port, os.R_OK | os.W_OK))},
        "whisper": TRANSCRIBER.status(cfg),
        "gemma": gemma_status(cfg),
        "numpy": np is not None,
        "job": job,
        "walks": walks,
        "week": week,
        "weather": weather_state(cfg),
        "config": cfg,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "FieldNotes/1.0"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            data = (HERE / "ui.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif url.path.startswith("/fonts/"):
            name = url.path[len("/fonts/"):]
            try:
                if not re.fullmatch(r"[a-z0-9-]+\.woff2", name):
                    raise ValueError
                data = (HERE / "fonts" / name).read_bytes()
            except (ValueError, OSError):
                return self._json({"error": "not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "font/woff2")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        elif url.path == "/api/pincode":
            try:
                self._json({"areas": pincode_areas(parse_qs(url.query).get("pin", [""])[0])})
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:
                self._json({"error": "Couldn't look up that PIN code (%s)." % e}, 502)
        elif url.path == "/api/state":
            self._json(state_payload())
        elif url.path.startswith("/audio/"):
            parts = [unquote(p) for p in url.path.split("/")[2:]]
            try:
                if len(parts) != 2 or not re.fullmatch(r"note_\d+\.wav", parts[1]):
                    raise ValueError
                f = walk_dir(parts[0]) / parts[1]
                data = f.read_bytes()
            except (ValueError, OSError):
                return self._json({"error": "not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/upload":
                batch = q.get("batch", "")
                if not re.fullmatch(r"[A-Za-z0-9-]{8,40}", batch):
                    raise ValueError("bad batch")
                d = INBOX / batch
                d.mkdir(parents=True, exist_ok=True)
                name = os.path.basename(q.get("name", "note"))[:80] or "note"
                stored = uuid.uuid4().hex + Path(name).suffix.lower()[:6]
                (d / stored).write_bytes(self._body())
                meta_p = d / "meta.json"
                with _store_lock:
                    meta = json.loads(meta_p.read_text()) if meta_p.exists() else []
                    meta.append({"name": name, "stored": stored, "mtime": int(q.get("mtime", "0") or 0)})
                    meta_p.write_text(json.dumps(meta))
                return self._json({"ok": True})

            body = json.loads(self._body() or b"{}")
            if url.path == "/api/import":
                if not find_recorder():
                    raise ValueError("The recorder isn't plugged in.")
                enqueue("Importing from the recorder", job_import_recorder, bool(body.get("clear", True)))
            elif url.path == "/api/upload-done":
                if not re.fullmatch(r"[A-Za-z0-9-]{8,40}", body["batch"]):
                    raise ValueError("bad batch")
                enqueue("Adding notes", job_process_upload, body["batch"])
            elif url.path == "/api/merge":
                merge_walks(body["into"], body["from"])
            elif url.path == "/api/transcribe":
                enqueue("Transcribing", job_transcribe, body["walk"])
            elif url.path == "/api/entry":
                enqueue("Writing the journal entry", job_entry, body["walk"])
            elif url.path == "/api/week":
                enqueue("Writing the weekly recap", job_week)
            elif url.path == "/api/transcript":
                walk = load_walk(body["walk"])
                for n in walk["notes"]:
                    if n["file"] == body["file"]:
                        n["transcript"] = str(body["text"])
                save_walk(walk)
            elif url.path == "/api/delete-walk":
                shutil.rmtree(walk_dir(body["walk"]), ignore_errors=True)
            elif url.path == "/api/config":
                cfg = load_config()
                for k in DEFAULTS:
                    if k in body:
                        cfg[k] = body[k]
                save_config(cfg)
            elif url.path == "/api/dismiss":
                job_update(error="", done="")
            elif url.path == "/api/quit":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            else:
                return self._json({"error": "not found"}, 404)
            self._json({"ok": True})
        except (ValueError, KeyError, OSError) as e:
            self._json({"error": str(e) or "bad request"}, 400)


def open_window(url):
    """A chromeless app window if Chromium/Chrome/Brave is installed, else the default browser."""
    for b in ("chromium", "google-chrome-stable", "google-chrome", "brave", "brave-browser",
              "microsoft-edge-stable", "vivaldi-stable"):
        exe = shutil.which(b)
        if exe:
            subprocess.Popen([exe, "--app=" + url, "--new-window"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            return
    webbrowser.open(url)


def main():
    load_env_file()
    url = "http://127.0.0.1:%d/" % PORT
    WALKS.mkdir(parents=True, exist_ok=True)
    INBOX.mkdir(parents=True, exist_ok=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print("Field Notes is already running; opening it.")
        if "--no-window" not in sys.argv:
            open_window(url)
        return
    threading.Thread(target=worker, daemon=True).start()
    print("Field Notes running at", url)
    if "--no-window" not in sys.argv:
        open_window(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
