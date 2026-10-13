#!/usr/bin/env python3
"""Auto dubbing Indonesia -> English. Sepenuhnya lokal untuk STT & TTS.

Pipeline:
  video -> ffmpeg (extract audio 16k mono)
        -> faster-whisper LOKAL (segments + timestamp, tanpa API)
        -> LLM (translate, dubbing-friendly)   <- satu-satunya panggilan cloud
        -> Piper TTS LOKAL (per segmen, tanpa API)
        -> ffmpeg (trim silence, shorten text / speed up if too long)
        -> assemble timeline -> mux ke video (extend video jika audio lebih panjang)

Kenapa lokal semua:
  - Edge TTS   : 403, Microsoft memblokir IP datacenter runner GitHub Actions.
  - Groq TTS   : canopylabs/orpheus-v1-english tier gratis RPM=10, RPD=100.
                 Video 92 segmen menghabiskan kuota harian dalam satu run.
  - Groq STT   : whisper-large-v3 RPM=20, RPD=2000 + butuh upload audio.
  Piper (onnxruntime) dan faster-whisper (CTranslate2) jalan di CPU runner:
  tanpa API key, tanpa kuota, tanpa rate limit, tidak bisa diblokir per IP.

Yang masih butuh jaringan hanya LLM penerjemah (LLM_API_KEY / LLM_BASE_URL)
dan download model sekali (di-cache antar run).
"""
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
import wave
from pathlib import Path

from openai import OpenAI

# faster_whisper dan piper diimpor lazy di dalam fungsinya masing-masing:
# ctranslate2 & onnxruntime berat (ratusan MB), dan tidak perlu dimuat hanya
# untuk membaca config atau menjalankan bagian lain dari script.


# ----------------------------------------------------------------- config
def env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    return v or default


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


VIDEO = Path(env("VIDEO_FILE"))
WORK = Path("work")
OUT = Path("output")

LLM_KEY = env("LLM_API_KEY")
LLM_BASE = env("LLM_BASE_URL")  # kosong = default OpenAI
LLM_MODEL = env("LLM_MODEL", "gpt-4o-mini")

MAX_SPEEDUP = max(1.0, min(float(env("MAX_SPEEDUP", "1.5")), 2.0))
KEEP_ORIG = float(env("KEEP_ORIGINAL_VOLUME", "0"))
BURN_SUBS = env_bool("BURN_SUBTITLES", True)
SUB_FONT_SIZE = env("SUBTITLE_FONT_SIZE")  # kosong = otomatis (landscape 16, portrait 12)
CPS = float(env("CHARS_PER_SEC", "14"))  # estimasi karakter/detik TTS inggris
SHORTEN_RATIO = 1.2  # jika audio > slot * 1.2 -> minta LLM mempersingkat dulu
SR = 24000  # sample rate audio dubbing

# --- STT: faster-whisper lokal ------------------------------------------
STT_MODEL = env("STT_MODEL", "large-v3-turbo")
STT_DEVICE = env("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = env("STT_COMPUTE_TYPE", "int8")
STT_THREADS = max(0, int(env("STT_THREADS", "0")))       # 0 = auto (semua core)
STT_BEAM_SIZE = max(1, int(env("STT_BEAM_SIZE", "5")))
STT_LANGUAGE = env("STT_LANGUAGE", "id")                 # kosong = auto-detect
# vad_filter=False memberi segmen lebih pendek (median 5.0s, max 11.2s vs 6.2s/13.7s
# pada clip uji 90 detik) sehingga lebih mudah pas di slot dubbing, dan sedikit
# lebih cepat. Nyalakan kalau video punya banyak hening dan muncul halusinasi.
STT_VAD = env_bool("STT_VAD_FILTER", False)
STT_MODEL_DIR = Path(env("STT_MODEL_DIR", ".whisper-models"))

# --- TTS lokal: piper (default) | kokoro --------------------------------
# Keduanya jalan di CPU runner, tanpa API key / kuota / rate limit.
#   piper  : RTF ~0.23x di 2 vCPU. Cepat, ringan (63 MB voice).
#   kokoro : RTF ~1.37x di 2 vCPU (~6x lebih lambat), tapi prosodinya lebih
#            natural. 82M parameter, model int8 92 MB + voices 28 MB.
TTS_PROVIDER = env("TTS_PROVIDER", "piper").lower()

PIPER_VOICE = env("TTS_PIPER_VOICE", "en_US-lessac-medium")
PIPER_DIR = Path(env("TTS_PIPER_DIR", ".piper-voices"))
PIPER_HF_BASE = env("TTS_PIPER_HF_BASE",
                    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0")
# length_scale > 1 = lebih lambat, < 1 = lebih cepat. Kosong = default model.
PIPER_LENGTH_SCALE = env("TTS_PIPER_LENGTH_SCALE")

# --- Kokoro-82M (opsi) ---------------------------------------------------
KOKORO_VOICE = env("TTS_KOKORO_VOICE", "af_bella")
KOKORO_DIR = Path(env("TTS_KOKORO_DIR", ".kokoro-models"))
KOKORO_LANG = env("TTS_KOKORO_LANG", "en-us")
KOKORO_SPEED = float(env("TTS_KOKORO_SPEED", "1.0"))
KOKORO_MODEL_FILE = env("TTS_KOKORO_MODEL_FILE", "kokoro-v1.0.int8.onnx")
KOKORO_VOICES_FILE = env("TTS_KOKORO_VOICES_FILE", "voices-v1.0.bin")
KOKORO_BASE_URL = env(
    "TTS_KOKORO_BASE_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0")
# Kokoro mengalokasikan memori seiring panjang input (terukur: satu buffer conv
# 157 MB untuk teks ~1000 karakter). Teks di atas batas ini dipecah per kalimat
# lalu audio-nya disambung lagi di level array, bukan lewat ffmpeg.
KOKORO_MAX_CHARS = max(100, int(env("TTS_KOKORO_MAX_CHARS", "500")))

# Kokoro ~6x lebih lambat dari Piper, jadi butuh timeout lebih longgar.
_DEFAULT_TIMEOUT = "240" if TTS_PROVIDER == "kokoro" else "90"
TTS_TIMEOUT = float(env("TTS_TIMEOUT", _DEFAULT_TIMEOUT))
_conc_raw = env("TTS_CONCURRENCY")
if _conc_raw:
    TTS_CONCURRENCY = max(1, int(_conc_raw))
else:
    # Default sadar-provider, berdasarkan pengukuran di runner 2 vCPU
    # (103 segmen, ~137 detik audio):
    #   kokoro  concurrency=4 -> RTF ~2.75x   <- 3x LEBIH LAMBAT
    #   kokoro  concurrency=2 -> RTF  0.96x
    #   kokoro  concurrency=1 -> RTF  0.92x
    #   piper   concurrency=4 -> RTF  0.165x
    # onnxruntime sudah memakai semua core dalam satu sesi, jadi menjalankan
    # 4 sesi Kokoro bersamaan di 2 vCPU = oversubscription dan thrashing.
    # Piper jauh lebih ringan sehingga paralelisme tetap menguntungkan.
    TTS_CONCURRENCY = 4 if TTS_PROVIDER == "piper" else 2
TTS_TRIES = max(1, int(env("TTS_TRIES", env("PIPER_TRIES", "3"))))
TTS_BACKOFF = float(env("TTS_BACKOFF", env("PIPER_BACKOFF", "2")))

llm = None            # di-init di main()
_STT = None           # model faster-whisper, di-load sekali
_STT_LOCK = threading.Lock()
_PIPER_VOICE = None   # model Piper, di-load sekali
_PIPER_LOCK = threading.Lock()
_KOKORO = None        # instance kokoro_onnx.Kokoro, di-load sekali
_KOKORO_LOCK = threading.Lock()


# ----------------------------------------------------------------- utils
class FFmpegError(RuntimeError):
    """Kegagalan ffmpeg yang dilempar sebagai exception biasa.

    Wajib dipakai di kode yang jalan dalam thread/async (to_wav): `sys.exit()`
    di dalam thread hanya melempar SystemExit yang tidak tertangkap
    `except Exception`, sehingga menembus logika retry.
    """


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"Command gagal: {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    return p.stdout


def run_cmd(cmd):
    """Seperti run(), tapi raise RuntimeError (bukan sys.exit) -> aman di thread."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise FFmpegError(
            f"Command gagal: {' '.join(map(str, cmd))}\n{p.stderr[-1500:]}")
    return p.stdout


def probe_duration(path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def has_audio(path) -> bool:
    out = run(["ffprobe", "-v", "error", "-select_streams", "a",
               "-show_entries", "stream=index", "-of", "csv=p=0", path])
    return bool(out.strip())


def probe_size(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path])
    w, h = out.strip().split("x")[:2]
    return int(w), int(h)


def wav_duration(path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------- 1. STT (lokal)
def extract_audio() -> Path:
    """Video -> WAV 16 kHz mono.

    faster-whisper membaca audio lewat PyAV, tapi kita seragamkan ke 16 kHz mono
    lebih dulu supaya deterministik dan sama persis dengan yang dulu dikirim ke
    Groq. Tidak ada lagi pemecahan per 20 menit: itu cuma karena Groq membatasi
    ukuran file 24 MB, dan Whisper lokal tidak punya batas semacam itu (memori
    terikat pada jendela 30 detik, bukan panjang file).
    """
    out = WORK / "audio.wav"
    run(["ffmpeg", "-y", "-i", VIDEO, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", out])
    return out


def load_stt_model():
    """Load (dan download kalau perlu) model faster-whisper. Sekali per proses."""
    global _STT
    if _STT is None:
        with _STT_LOCK:
            if _STT is None:                      # double-checked locking
                from faster_whisper import WhisperModel
                STT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                try:
                    _STT = WhisperModel(
                        STT_MODEL, device=STT_DEVICE,
                        compute_type=STT_COMPUTE_TYPE,
                        cpu_threads=STT_THREADS,
                        download_root=str(STT_MODEL_DIR))
                except Exception as e:  # noqa: BLE001
                    sys.exit(
                        f"[stt] gagal memuat model '{STT_MODEL}': "
                        f"{type(e).__name__}: {e}\n"
                        f"  - Nama model valid: tiny, base, small, medium, large-v3,\n"
                        f"    large-v3-turbo, distil-large-v3 (atau path direktori CT2).\n"
                        f"  - Kalau download dari huggingface.co gagal, jalankan ulang\n"
                        f"    (model di-cache di {STT_MODEL_DIR}/).\n"
                        f"  - Kalau proses di-kill (OOM), pakai STT_COMPUTE_TYPE=int8\n"
                        f"    dan model lebih kecil, mis. STT_MODEL=small.")
                print(f"[stt] model {STT_MODEL} siap dalam {time.time() - t0:.1f}s "
                      f"(device={STT_DEVICE}, compute={STT_COMPUTE_TYPE}, "
                      f"threads={STT_THREADS or 'auto'})")
    return _STT


def merge_short(segs, min_dur=1.0, max_gap=0.4, max_len=12.0):
    """Gabung segmen yang terlalu pendek supaya TTS lebih natural."""
    out = []
    for s in segs:
        if out:
            p = out[-1]
            if ((p["end"] - p["start"]) < min_dur
                    and (s["start"] - p["end"]) < max_gap
                    and (p["end"] - p["start"]) < max_len):
                p["end"] = s["end"]
                p["src"] += " " + s["src"]
                continue
        out.append(dict(s))
    return out


def transcribe():
    """Transkripsi lokal. `segments` adalah generator: waktu diukur mengelilingi
    iterasinya, bukan pemanggilan transcribe()."""
    model = load_stt_model()
    audio = extract_audio()
    video_dur = probe_duration(VIDEO)
    print(f"[stt] {audio.name} ({audio.stat().st_size / 1e6:.1f} MB, "
          f"video {video_dur:.1f}s)")

    vad_params = {"min_silence_duration_ms": 400} if STT_VAD else None
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio),
        language=STT_LANGUAGE or None,
        task="transcribe",
        beam_size=STT_BEAM_SIZE,
        vad_filter=STT_VAD,
        vad_parameters=vad_params,
        condition_on_previous_text=True,
    )
    segs = []
    try:
        for s in segments:
            text = s.text.strip()
            if text:
                segs.append({"start": float(s.start), "end": float(s.end), "src": text})
    except (RuntimeError, MemoryError) as e:
        # OOM muncul di SINI (saat encode/inference), bukan saat load model.
        # ctranslate2 melaporkannya sebagai "mkl_malloc: failed to allocate memory".
        msg = f"{type(e).__name__}: {e}"
        hint = ""
        if "alloc" in msg.lower() or "memory" in msg.lower() or isinstance(e, MemoryError):
            hint = (
                "\n  Ini kehabisan RAM saat inference. Urutan yang paling menolong:\n"
                "    1. STT_THREADS=1        (tiap thread alokasikan buffer sendiri)\n"
                "    2. STT_COMPUTE_TYPE=int8 (sudah default; jangan pakai float32)\n"
                "    3. STT_MODEL=small lalu base (turbo ~3 GB, small ~1 GB, base ~0.5 GB)\n"
                f"  Konfigurasi sekarang: model={STT_MODEL} compute={STT_COMPUTE_TYPE} "
                f"threads={STT_THREADS or 'auto'}\n"
                f"  Sebagian hasil sudah didapat: {len(segs)} segmen sebelum gagal.")
        sys.exit(f"[stt] transkripsi gagal: {msg}{hint}")
    el = time.time() - t0
    segs.sort(key=lambda x: x["start"])
    print(f"[stt] selesai {el:.1f}s (RTF {el / max(video_dur, 0.1):.3f}x) | "
          f"bahasa={info.language} p={info.language_probability:.2f} | "
          f"{len(segs)} segmen mentah")
    merged = merge_short(segs)
    if len(merged) != len(segs):
        print(f"[stt] merge_short: {len(segs)} -> {len(merged)} segmen")
    return merged


def compute_slots(segs, video_dur):
    """Slot = waktu yang tersedia untuk tiap segmen (sampai segmen berikutnya mulai)."""
    for i, s in enumerate(segs):
        if i < len(segs) - 1:
            slot = segs[i + 1]["start"] - s["start"] - 0.05
        else:
            slot = max(video_dur - s["start"], s["end"] - s["start"])
        s["slot"] = max(slot, 0.4)


# ----------------------------------------------------------------- 2. LLM
SYS_TRANSLATE = (
    "You are a professional dubbing translator (Indonesian -> English). "
    "Translate each line into natural, spoken English suitable for voice-over. "
    "Keep the meaning and proper nouns. Do NOT exceed each line's max_chars: "
    "paraphrase or condense if needed. No emojis, no stage directions. "
    'Respond ONLY with JSON: {"items":[{"idx":<int>,"en":"<text>"}]} '
    "with one entry for every input idx."
)


def strip_fences(txt: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.M).strip()


def llm_chat(system: str, user: str) -> str:
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def translate(segs, batch_size=25):
    for b in range(0, len(segs), batch_size):
        batch = segs[b:b + batch_size]
        payload = [{"idx": b + k,
                    "indonesian": s["src"],
                    "max_chars": max(8, int(s["slot"] * CPS))}
                   for k, s in enumerate(batch)]
        result = None
        for attempt in range(3):
            try:
                raw = llm_chat(SYS_TRANSLATE, json.dumps(payload, ensure_ascii=False))
                items = json.loads(strip_fences(raw))["items"]
                result = {int(it["idx"]): str(it["en"]).strip() for it in items}
                if all((b + k) in result for k in range(len(batch))):
                    break
                result = None
            except Exception as e:  # noqa: BLE001
                print(f"[llm] parse error attempt {attempt + 1}: {e}")
            time.sleep(2)
        if result is None:
            sys.exit("Gagal menerjemahkan batch dari LLM")
        for k, s in enumerate(batch):
            s["en"] = result[b + k]
        print(f"[llm] translated {min(b + batch_size, len(segs))}/{len(segs)}")


def shorten(text: str, target_chars: int) -> str:
    system = ("Rewrite the English dubbing line to be shorter while keeping the core meaning. "
              f"Maximum {target_chars} characters. Return ONLY the rewritten line.")
    try:
        out = llm_chat(system, text).strip().strip('"')
        return out or text
    except Exception as e:  # noqa: BLE001
        print(f"[llm] shorten gagal: {e}")
        return text


# ----------------------------------------------------------------- 3. TTS (lokal) + fit
TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse")


def to_wav(src: Path, dst: Path, tempo: float = 1.0):
    """audio (wav Piper) -> wav mono 24k, trim silence di ujung, opsional percepat.

    Dipanggil dari asyncio.to_thread, jadi pakai run_cmd (raise) bukan run (sys.exit).
    Piper menghasilkan 22050 Hz; di sini di-resample ke SR=24000 supaya seragam
    dengan timeline assemble().
    """
    f = TRIM + (f",atempo={tempo:.4f}" if tempo > 1.0 else "")
    run_cmd(["ffmpeg", "-y", "-i", src, "-af", f, "-ar", SR, "-ac", "1",
             "-c:a", "pcm_s16le", dst])
    if wav_duration(dst) < 0.05:  # fallback bila trim menghabiskan semua audio
        f = f"atempo={tempo:.4f}" if tempo > 1.0 else "anull"
        run_cmd(["ffmpeg", "-y", "-i", src, "-af", f, "-ar", SR, "-ac", "1",
                 "-c:a", "pcm_s16le", dst])


def _download_file(url: str, dst: Path, tries: int = 3):
    """Download atomik (tulis ke .part lalu rename) dengan retry.

    Dipakai bersama oleh Piper dan Kokoro. Atomic rename penting supaya file
    setengah jadi tidak dianggap cache valid oleh run berikutnya.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.parent / (dst.name + ".part")
    for attempt in range(tries):
        try:
            print(f"[model] download {url}")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            print(f"[model] {dst.name}: {dst.stat().st_size / 1e6:.1f} MB")
            return dst
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            if attempt == tries - 1:
                raise RuntimeError(f"gagal download {dst.name} dari {url}: {e}") from e
            time.sleep(3 * (attempt + 1))
    return dst


def _piper_ensure_model(name: str) -> Path:
    """Pastikan model suara Piper tersedia; download dari HuggingFace kalau belum.

    `name` boleh "en_US-lessac-medium" (di-download ke PIPER_DIR) atau path
    langsung ke file .onnx. File konfigurasi .onnx.json di-load otomatis oleh
    PiperVoice.load() dari samping file model.
    """
    if name.endswith(".onnx"):
        direct = Path(name)
        if not direct.is_file():
            raise RuntimeError(f"Model Piper tidak ditemukan: {direct}")
        return direct

    parts = name.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"TTS_PIPER_VOICE '{name}' tidak valid. Format: <lang>_<REGION>-<nama>-"
            f"<kualitas>, contoh en_US-lessac-medium")
    lang, vname, quality = parts
    family = lang.split("_")[0]
    base = f"{PIPER_HF_BASE}/{family}/{lang}/{vname}/{quality}/{name}"

    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    onnx = PIPER_DIR / f"{name}.onnx"
    _download_file(base + ".onnx.json", PIPER_DIR / f"{name}.onnx.json")
    _download_file(base + ".onnx", onnx)
    return onnx


def _piper_get_voice():
    """Load model Piper sekali per proses. Thread-safe: dipanggil dari to_thread."""
    global _PIPER_VOICE
    if _PIPER_VOICE is None:
        with _PIPER_LOCK:
            if _PIPER_VOICE is None:              # double-checked locking
                from piper import PiperVoice      # lazy: onnxruntime berat
                onnx = _piper_ensure_model(PIPER_VOICE)
                t0 = time.time()
                _PIPER_VOICE = PiperVoice.load(str(onnx))
                print(f"[piper] voice {PIPER_VOICE} siap dalam {time.time() - t0:.1f}s "
                      f"({_PIPER_VOICE.config.sample_rate} Hz)")
    return _PIPER_VOICE


def _piper_once(text: str, out: Path):
    """Sintesis lokal -> WAV. Tanpa API key, tanpa rate limit, tanpa blokir IP."""
    from piper import SynthesisConfig
    voice = _piper_get_voice()
    cfg = None
    if PIPER_LENGTH_SCALE:
        try:
            cfg = SynthesisConfig(length_scale=float(PIPER_LENGTH_SCALE))
        except ValueError:
            print(f"[piper] TTS_PIPER_LENGTH_SCALE '{PIPER_LENGTH_SCALE}' "
                  "bukan angka, diabaikan")
    tmp = out.parent / (out.stem + ".piper.part.wav")
    with wave.open(str(tmp), "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=cfg)
    if not tmp.exists() or tmp.stat().st_size < 100:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Piper menghasilkan audio kosong")
    tmp.replace(out)


# --- provider: Kokoro-82M (opsi) ----------------------------------------
def kokoro_split(text: str, limit: int = None):
    """Pecah teks per batas kalimat supaya tidak melewati KOKORO_MAX_CHARS.

    Kokoro tidak punya batas karakter keras seperti Orpheus (batasnya 510 fonem
    dan dipecah sendiri oleh kokoro-onnx), TAPI alokasi memorinya tumbuh seiring
    panjang input. Memecah di sini membuat pemakaian RAM tiap panggilan terikat
    dan bisa diprediksi.
    """
    limit = limit or KOKORO_MAX_CHARS
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text] if text else []
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?;:])\s+", text):
        sent = sent.strip()
        while len(sent) > limit:
            cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if not sent:
            continue
        if cur and len(cur) + 1 + len(sent) > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts or [text[:limit]]


def _kokoro_ensure_model():
    """Pastikan model + voices Kokoro tersedia di KOKORO_DIR."""
    low = KOKORO_MODEL_FILE.lower()
    if "fp16" in low or "q8f16" in low or "q4f16" in low:
        print(f"[kokoro] PERINGATAN: '{KOKORO_MODEL_FILE}' memakai bobot fp16 dan "
              "TERUKUR SEGFAULT di onnxruntime CPU (bukan exception Python, jadi "
              "tidak bisa ditangkap). Pakai kokoro-v1.0.int8.onnx (default) atau "
              "kokoro-v1.0.onnx (fp32).")
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    voices = _download_file(f"{KOKORO_BASE_URL}/{KOKORO_VOICES_FILE}",
                            KOKORO_DIR / KOKORO_VOICES_FILE)
    model = _download_file(f"{KOKORO_BASE_URL}/{KOKORO_MODEL_FILE}",
                           KOKORO_DIR / KOKORO_MODEL_FILE)
    return model, voices


def _kokoro_get():
    """Load Kokoro sekali per proses. Thread-safe: dipanggil dari to_thread."""
    global _KOKORO
    if _KOKORO is None:
        with _KOKORO_LOCK:
            if _KOKORO is None:                 # double-checked locking
                from kokoro_onnx import Kokoro   # lazy: tidak perlu untuk piper
                model, voices = _kokoro_ensure_model()
                t0 = time.time()
                inst = Kokoro(str(model), str(voices))
                avail = inst.get_voices()
                # Validasi SEBELUM assign ke _KOKORO. Kalau assign lebih dulu lalu
                # raise, retry berikutnya melihat _KOKORO sudah terisi dan melewati
                # validasi ini -- yang muncul malah ValueError internal kokoro_onnx
                # ("Voice X not found in available voices") tanpa daftar pilihan.
                if KOKORO_VOICE not in avail:
                    raise RuntimeError(
                        f"Voice Kokoro '{KOKORO_VOICE}' tidak ada. "
                        f"{len(avail)} pilihan tersedia, mis: "
                        f"{', '.join(sorted(avail)[:20])} ...")
                _KOKORO = inst
                print(f"[kokoro] model siap dalam {time.time() - t0:.1f}s "
                      f"(voice {KOKORO_VOICE}, {len(avail)} voice tersedia, "
                      f"lang={KOKORO_LANG}, speed={KOKORO_SPEED})")
    return _KOKORO


def _kokoro_once(text: str, out: Path):
    """Sintesis Kokoro -> WAV 24 kHz. Output-nya sudah persis SR pipeline."""
    import numpy as np
    k = _kokoro_get()
    pieces = []
    for part in kokoro_split(text):
        audio, sr = k.create(part, voice=KOKORO_VOICE,
                             speed=KOKORO_SPEED, lang=KOKORO_LANG)
        if audio is None or len(audio) == 0:
            raise RuntimeError(f"Kokoro menghasilkan audio kosong untuk: {part!r}")
        pieces.append((np.asarray(audio, dtype=np.float32), sr))
    if not pieces:
        raise RuntimeError("Kokoro tidak menghasilkan audio")
    # gabung di level array (bukan ffmpeg): semua piece sr-nya sama
    sr = pieces[0][1]
    merged = pieces[0][0] if len(pieces) == 1 else np.concatenate([p[0] for p in pieces])
    pcm = (np.clip(merged, -1.0, 1.0) * 32767).astype(np.int16)
    tmp = out.parent / (out.stem + ".kokoro.part.wav")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    if not tmp.exists() or tmp.stat().st_size < 100:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Kokoro menghasilkan audio kosong")
    tmp.replace(out)


def _tts_impl():
    """Fungsi sintesis untuk provider yang dipilih."""
    if TTS_PROVIDER == "kokoro":
        return _kokoro_once
    if TTS_PROVIDER != "piper":
        raise RuntimeError(
            f"TTS_PROVIDER '{TTS_PROVIDER}' tidak dikenal. Pilihan: piper, kokoro.")
    return _piper_once


async def tts(text: str, out: Path, sem: asyncio.Semaphore) -> str:
    """Sintesis satu segmen ke `out`. Return nama provider kalau berhasil."""
    impl = _tts_impl()
    async with sem:
        last = None
        for attempt in range(TTS_TRIES):
            try:
                # wait_for: sintesis yang menggantung tidak boleh memblokir selamanya
                await asyncio.wait_for(asyncio.to_thread(impl, text, out),
                                       timeout=TTS_TIMEOUT)
                if out.exists() and out.stat().st_size > 0:
                    return TTS_PROVIDER
                last = RuntimeError("hasil TTS kosong")
            except asyncio.TimeoutError:
                last = RuntimeError(f"timeout >{TTS_TIMEOUT:.0f}s")
            except Exception as e:  # noqa: BLE001
                last = e
            if attempt < TTS_TRIES - 1:
                await asyncio.sleep(TTS_BACKOFF * (attempt + 1))
        raise RuntimeError(f"{TTS_PROVIDER} TTS gagal untuk: {text!r} ({last})")


async def preflight_tts():
    """Tes TTS SEBELUM menghabiskan waktu STT + kuota LLM translate.

    Tanpa ini, run baru tahu TTS-nya mati setelah beberapa menit dan setelah
    puluhan segmen terjemahan terbuang (yang terjadi sebelum preflight ada).
    """
    probe = WORK / "_preflight.wav"
    _tts_impl()   # validasi TTS_PROVIDER lebih awal, pesan error jelas
    sem = asyncio.Semaphore(1)
    t0 = time.time()
    try:
        await asyncio.wait_for(
            tts("Preflight check for the dubbing pipeline.", probe, sem),
            timeout=TTS_TIMEOUT * 2 + 60)
        detail = (f"voice {PIPER_VOICE}" if TTS_PROVIDER == "piper"
                  else f"voice {KOKORO_VOICE}, lang {KOKORO_LANG}")
        print(f"[preflight] {TTS_PROVIDER} TTS OK dalam {time.time() - t0:.1f}s ({detail})")
    except Exception as e:  # noqa: BLE001
        if TTS_PROVIDER == "kokoro":
            hint = ("  - Voice Kokoro harus salah satu dari 54 nama bawaan, mis.\n"
                    "    af_bella, af_sarah, am_adam, am_michael, bm_george.\n"
                    "  - Model HARUS varian int8/fp32. Varian fp16 (q8f16/q4f16/fp16)\n"
                    "    terukur SEGFAULT di onnxruntime CPU dan tidak bisa ditangkap.\n"
                    "  - Kalau 'Failed to allocate memory': teks terlalu panjang atau\n"
                    "    RAM kurang. Turunkan TTS_KOKORO_MAX_CHARS.")
        else:
            hint = ("  - Cek nama voice (format en_US-lessac-medium) dan akses ke\n"
                    "    huggingface.co dari runner.\n"
                    "  - Kalau proses di-kill tanpa pesan: onnxruntime kehabisan memori.")
        sys.exit(f"[preflight] {TTS_PROVIDER} TTS gagal: {type(e).__name__}: {e}\n{hint}")
    finally:
        probe.unlink(missing_ok=True)


def tts_signature(text: str) -> str:
    """Identitas lengkap sebuah hasil sintesis: provider + voice + parameter + teks.

    WAJIB memuat provider dan voice. Kalau tag hanya berisi teks, mengganti
    tts_provider atau voice akan diam-diam memakai ulang audio provider lama dari
    cache -- video hasilnya bersuara provider yang salah tanpa peringatan apa pun,
    dan pengujian provider baru akan "lulus" padahal tidak pernah dijalankan.

    Perubahan format ini juga otomatis menginvalidasi cache lama (yang hanya
    berisi teks), dan itu arah yang aman: regenerate, bukan pakai yang basi.
    """
    if TTS_PROVIDER == "kokoro":
        cfg = f"kokoro|{KOKORO_VOICE}|{KOKORO_LANG}|{KOKORO_SPEED}"
    else:
        cfg = f"piper|{PIPER_VOICE}|{PIPER_LENGTH_SCALE or 'default'}"
    return f"{cfg}|{text}"


async def build_clip(i: int, seg: dict, sem: asyncio.Semaphore):
    aud = WORK / f"tts_{i:04d}.audio"   # wav dari Piper/Kokoro; ffmpeg sniff isi file
    wav = WORK / f"clip_{i:04d}.wav"
    tag = WORK / f"tts_{i:04d}.txt"     # isi = tts_signature() -> invalidasi cache
    slot = seg["slot"]
    text = seg["en"]

    def cached(t: str) -> bool:
        return (aud.exists() and aud.stat().st_size > 0
                and tag.exists()
                and tag.read_text(encoding="utf-8") == tts_signature(t))

    if cached(text):
        prov = "cache"
    else:
        prov = await tts(text, aud, sem)
        tag.write_text(tts_signature(text), encoding="utf-8")
    await asyncio.to_thread(to_wav, aud, wav)
    dur = wav_duration(wav)
    info = prov

    # 1) Terlalu panjang -> minta LLM mempersingkat, lalu generate ulang
    if dur > slot * SHORTEN_RATIO:
        target = max(6, int(len(text) * slot * 1.05 / dur))
        new_text = await asyncio.to_thread(shorten, text, target)
        if new_text and new_text != text:
            text = new_text
            if not cached(text):
                prov = await tts(text, aud, sem)
                tag.write_text(tts_signature(text), encoding="utf-8")
            await asyncio.to_thread(to_wav, aud, wav)
            dur = wav_duration(wav)
            info = f"{prov}+shortened"

    # 2) Masih lebih panjang dari slot -> percepat (atempo), dibatasi MAX_SPEEDUP
    tempo = 1.0
    if dur > slot:
        tempo = min(dur / slot, MAX_SPEEDUP)
        if tempo > 1.02:
            await asyncio.to_thread(to_wav, aud, wav, tempo)
            dur = wav_duration(wav)
            info += f"+x{tempo:.2f}"

    # 3) Jika masih > slot (melewati MAX_SPEEDUP) -> akan mendorong segmen berikutnya
    if dur > slot + 0.05:
        info += " OVERFLOW"

    seg["en"] = text
    print(f"[tts] #{i:04d} slot={slot:.2f}s dur={dur:.2f}s {info}")
    return wav, dur


async def build_all(segs):
    sem = asyncio.Semaphore(TTS_CONCURRENCY)
    # return_exceptions: satu segmen rusak tidak boleh membuang puluhan segmen
    # yang sudah berhasil (perilaku lama: gather abort -> semua kerja hilang).
    res = await asyncio.gather(*(build_clip(i, s, sem) for i, s in enumerate(segs)),
                               return_exceptions=True)
    failed = [(i, r) for i, r in enumerate(res) if isinstance(r, BaseException)]
    if failed:
        OUT.mkdir(exist_ok=True)
        detail = [{"idx": i,
                   "en": segs[i].get("en"),
                   "slot": round(segs[i].get("slot", 0), 2),
                   "error": f"{type(r).__name__}: {r}"} for i, r in failed]
        (OUT / "tts_failures.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[tts] GAGAL {len(failed)}/{len(segs)} segmen")
        for i, r in failed[:8]:
            print(f"  #{i:04d}: {type(r).__name__}: {str(r)[:180]}")
        if len(failed) > 8:
            print(f"  ... dan {len(failed) - 8} lainnya")
        print("  detail lengkap: output/tts_failures.json")
        raise SystemExit(2)
    return res


# ----------------------------------------------------------------- 4. assemble + mux
def assemble(segs, clips, video_dur):
    """Susun timeline audio. Segmen ditaruh di start-nya; jika segmen sebelumnya
    meluber, segmen ini digeser tepat setelahnya (tidak ada audio yang terpotong)."""
    out_path = OUT / "dub.wav"
    placed = []
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        cursor = 0  # dalam sample
        for seg, (wav, _dur) in zip(segs, clips):
            start = int(seg["start"] * SR)
            if start > cursor:
                w.writeframes(b"\x00\x00" * (start - cursor))
                cursor = start
            with wave.open(str(wav)) as r:
                frames = r.readframes(r.getnframes())
            n = len(frames) // 2
            placed.append({"start": cursor / SR, "end": (cursor + n) / SR, "text": seg["en"]})
            w.writeframes(frames)
            cursor += n
        end = int(video_dur * SR)
        if end > cursor:
            w.writeframes(b"\x00\x00" * (end - cursor))
            cursor = end
    return out_path, cursor / SR, placed


def subtitle_style() -> str:
    w, h = probe_size(VIDEO)
    size = SUB_FONT_SIZE or ("12" if h > w else "16")
    return (f"FontName=DejaVu Sans,FontSize={size},Bold=1,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,Alignment=2,MarginV=18")


def mux(dub_wav: Path, dub_dur: float, video_dur: float, srt: Path = None):
    out = OUT / "dubbed.mp4"
    extend = dub_dur - video_dur  # audio lebih panjang dari video?
    chain = []

    if extend > 0.3:
        print(f"[mux] audio lebih panjang {extend:.2f}s -> video diperpanjang (freeze frame terakhir)")
        chain.append(f"tpad=stop_mode=clone:stop_duration={extend:.3f}")
    if srt is not None:
        print("[mux] burn subtitle bahasa Inggris ke video")
        chain.append(f"subtitles={srt}:force_style='{subtitle_style()}'")

    fc = []
    if chain:  # filter video -> wajib re-encode
        fc.append("[0:v]" + ",".join(chain) + "[v]")
        vmap = "[v]"
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        vmap = "0:v:0"
        vcodec = ["-c:v", "copy"]

    if KEEP_ORIG > 0 and has_audio(VIDEO):
        fc.append(f"[0:a]volume={KEEP_ORIG}[bg];[bg][1:a]amix=inputs=2:duration=longest:normalize=0[a]")
        amap = "[a]"
    else:
        amap = "1:a:0"

    cmd = ["ffmpeg", "-y", "-i", VIDEO, "-i", dub_wav]
    if fc:
        cmd += ["-filter_complex", ";".join(fc)]
    cmd += ["-map", vmap, "-map", amap, *vcodec,
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    run(cmd)
    return out


def ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_cues(p: dict, width: int):
    """Pecah teks panjang jadi cue maks 2 baris; durasi dibagi proporsional."""
    text = " ".join(p["text"].split())
    lines = textwrap.wrap(text, width=width) or [text]
    groups = [lines[i:i + 2] for i in range(0, len(lines), 2)]
    total = sum(len(" ".join(g)) for g in groups) or 1
    dur = p["end"] - p["start"]
    cues, t = [], p["start"]
    for g in groups:
        share = len(" ".join(g)) / total * dur
        cues.append((t, t + share, "\n".join(g)))
        t += share
    return cues


def write_srt(placed) -> Path:
    w, h = probe_size(VIDEO)
    width = 26 if h > w else 42  # karakter per baris (portrait vs landscape)
    lines, n = [], 0
    for p in placed:
        for start, end, text in split_cues(p, width):
            n += 1
            lines += [str(n), f"{ts(start)} --> {ts(end)}", text, ""]
    path = OUT / "dubbed.en.srt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------- 5. cache STT+translate
STATE = WORK / "state.json"


def load_state(video_sha: str):
    """Kembalikan segmen hasil run sebelumnya kalau video & modelnya sama persis."""
    if not STATE.exists():
        return None
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[cache] state.json tidak bisa dibaca, hitung ulang ({e})")
        return None
    alasan = []
    if st.get("video_sha") != video_sha:
        alasan.append(f"video_sha {str(st.get('video_sha'))[:12]}.. != {video_sha[:12]}..")
    if st.get("stt_model") != STT_MODEL:
        alasan.append(f"stt_model {st.get('stt_model')!r} != {STT_MODEL!r}")
    if st.get("llm_model") != LLM_MODEL:
        alasan.append(f"llm_model {st.get('llm_model')!r} != {LLM_MODEL!r}")
    seg = st.get("segments")
    if not isinstance(seg, list) or not seg:
        alasan.append("segments kosong/bukan list")
    else:
        kosong = sum(1 for s in seg if not s.get("en"))
        if kosong:
            alasan.append(f"{kosong} segmen tanpa terjemahan 'en'")
    if not alasan:
        return seg
    print(f"[cache] state.json DITOLAK -> hitung ulang. Alasan: {'; '.join(alasan)}")
    return None


def save_state(video_sha: str, segs):
    STATE.write_text(json.dumps({
        "video_sha": video_sha,
        "stt_model": STT_MODEL,
        "llm_model": LLM_MODEL,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segs,
    }, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------- main
def main():
    global llm
    if not VIDEO.is_file():
        sys.exit(f"Video tidak ditemukan: {VIDEO}")
    if not LLM_KEY:
        sys.exit("LLM_API_KEY belum di-set")
    llm = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE or None)

    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    video_dur = probe_duration(VIDEO)
    print(f"[init] video {video_dur:.1f}s, max_speedup={MAX_SPEEDUP}")
    print(f"[init] STT lokal: model={STT_MODEL} device={STT_DEVICE} "
          f"compute={STT_COMPUTE_TYPE} vad={STT_VAD} lang={STT_LANGUAGE or 'auto'}")
    if TTS_PROVIDER == "kokoro":
        print(f"[init] TTS lokal: provider=kokoro voice={KOKORO_VOICE} "
              f"lang={KOKORO_LANG} speed={KOKORO_SPEED} concurrency={TTS_CONCURRENCY}")
        print(f"[init] catatan: Kokoro ~6x lebih lambat dari Piper "
              f"(RTF ~1.4 vs ~0.23 di 2 vCPU)")
    else:
        print(f"[init] TTS lokal: provider=piper voice={PIPER_VOICE} "
              f"concurrency={TTS_CONCURRENCY}")
    print(f"[init] LLM (satu-satunya cloud): model={LLM_MODEL} "
          f"base_url={LLM_BASE or '(default OpenAI)'}")

    video_sha = sha256_file(VIDEO)
    print(f"[init] sha256 video {video_sha[:16]}...")

    t_start = time.time()
    timing = {}          # nama fase -> detik, dicetak sebagai ringkasan di akhir
    _mark = [t_start]

    def lap(name):
        """Catat durasi fase sejak `lap` terakhir."""
        now = time.time()
        timing[name] = now - _mark[0]
        _mark[0] = now
        print(f"[waktu] {name:<22} {timing[name]:7.1f}s ({timing[name] / 60:.1f} mnt)")

    # 0) Cek TTS dulu: murah (~2 detik), hemat waktu STT & kuota LLM kalau mati.
    asyncio.run(preflight_tts())
    lap("preflight TTS")

    # 1-2) STT lokal + translate (di-cache antar run untuk video yang sama)
    segs = load_state(video_sha)
    if segs is not None:
        print(f"[cache] pakai ulang hasil STT+translate: {len(segs)} segmen "
              "(skip faster-whisper & LLM)")
        compute_slots(segs, video_dur)
        lap("STT+LLM (dari cache)")
    else:
        segs = transcribe()
        if not segs:
            sys.exit("Tidak ada ucapan terdeteksi")
        print(f"[stt] {len(segs)} segmen")
        lap("STT lokal")
        compute_slots(segs, video_dur)
        translate(segs)
        lap("LLM translate")
        save_state(video_sha, segs)

    # 3) TTS lokal
    clips = asyncio.run(build_all(segs))
    lap(f"TTS {TTS_PROVIDER}")

    # 4) assemble + mux
    dub_wav, dub_dur, placed = assemble(segs, clips, video_dur)
    print(f"[assemble] audio dubbing {dub_dur:.1f}s vs video {video_dur:.1f}s")

    srt = write_srt(placed)
    mux(dub_wav, dub_dur, video_dur, srt if BURN_SUBS else None)
    save_state(video_sha, segs)  # teks final (mungkin sudah di-shorten)
    (OUT / "segments.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    lap("assemble+mux+srt")
    total = time.time() - t_start

    # Ringkasan waktu per fase. Penting untuk membandingkan provider: tanpa ini
    # hanya total yang terlihat, padahal STT dan TTS bisa bertukar porsi.
    audio_dur = sum(c[1] for c in clips) if clips else 0.0
    print("\n[ringkasan waktu]")
    for name, sec in timing.items():
        print(f"  {name:<22} {sec:7.1f}s  {100 * sec / total:5.1f}%")
    print(f"  {'TOTAL':<22} {total:7.1f}s  ({total / 60:.1f} menit)")
    t_tts = timing.get(f"TTS {TTS_PROVIDER}", 0.0)
    if t_tts and audio_dur:
        print(f"  RTF TTS terukur        {t_tts / audio_dur:7.3f}x "
              f"({t_tts:.0f}s proses / {audio_dur:.0f}s audio, "
              f"{len(segs)} segmen)")
    print(f"[done] output/dubbed.mp4, dubbed.en.srt, dub.wav, segments.json "
          f"| {len(segs)} segmen | total {total:.0f}s ({total / 60:.1f} menit)")


if __name__ == "__main__":
    main()
