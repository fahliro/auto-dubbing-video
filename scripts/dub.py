#!/usr/bin/env python3
"""Auto dubbing Indonesia -> English.

Pipeline:
  video -> ffmpeg (extract audio) -> Groq Whisper (segments + timestamp)
        -> LLM (translate, dubbing-friendly) -> TTS (per segment)
        -> ffmpeg (trim silence, shorten text / speed up if too long)
        -> assemble timeline -> mux ke video (extend video jika audio masih lebih panjang)

TTS provider (env TTS_PROVIDER):
  groq  -> Groq Orpheus. Default: andal dari GitHub Actions karena Microsoft
           memblokir IP datacenter runner dengan 403. Butuh GROQ_API_KEY.
  auto  -> coba Edge TTS dulu; kalau kena 403, otomatis pindah ke Groq Orpheus
           untuk SEMUA segmen sisanya.
  edge  -> paksa Edge TTS saja (gagal keras kalau IP runner diblokir)
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

import requests
from openai import OpenAI

# edge_tts diimpor lazy di dalam _edge_once(): kalau TTS_PROVIDER=groq, run tidak
# boleh gagal hanya karena paket edge-tts bermasalah.


# ----------------------------------------------------------------- config
def env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    return v or default


VIDEO = Path(env("VIDEO_FILE"))
WORK = Path("work")
OUT = Path("output")

GROQ_KEY = env("GROQ_API_KEY")
STT_MODEL = env("STT_MODEL", "whisper-large-v3")

LLM_KEY = env("LLM_API_KEY")
LLM_BASE = env("LLM_BASE_URL")  # kosong = default OpenAI
LLM_MODEL = env("LLM_MODEL", "gpt-4o-mini")

VOICE = env("TTS_VOICE", "en-US-AndrewNeural")
RATE = env("TTS_RATE", "+0%")  # rate dasar Edge TTS, mis. "+5%"
MAX_SPEEDUP = max(1.0, min(float(env("MAX_SPEEDUP", "1.5")), 2.0))
KEEP_ORIG = float(env("KEEP_ORIGINAL_VOLUME", "0"))
BURN_SUBS = env("BURN_SUBTITLES", "true").lower() in ("1", "true", "yes")
SUB_FONT_SIZE = env("SUBTITLE_FONT_SIZE")  # kosong = otomatis (landscape 16, portrait 12)
CPS = float(env("CHARS_PER_SEC", "14"))  # estimasi karakter/detik TTS inggris
SHORTEN_RATIO = 1.2  # jika audio > slot * 1.2 -> minta LLM mempersingkat dulu
SR = 24000  # sample rate audio dubbing

# --- TTS provider -------------------------------------------------------
TTS_PROVIDER = env("TTS_PROVIDER", "piper").lower()    # piper | groq | auto | edge
TTS_TIMEOUT = float(env("TTS_TIMEOUT", "90"))           # detik, per percobaan
TTS_CONCURRENCY = max(1, int(env("TTS_CONCURRENCY", "4")))
EDGE_TRIES = max(1, int(env("EDGE_TRIES", "3")))        # 403 = permanen, jangan 5x
GROQ_TRIES = max(1, int(env("GROQ_TRIES", "4")))
PIPER_TRIES = max(1, int(env("PIPER_TRIES", "2")))     # lokal, jarang gagal

GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
GROQ_TTS_MODEL = env("TTS_GROQ_MODEL", "canopylabs/orpheus-v1-english")
GROQ_TTS_VOICE = env("TTS_GROQ_VOICE", "troy")
# Orpheus: input maks 200 karakter. Pakai 190 supaya aman dari selisih hitung.
GROQ_CHAR_LIMIT = max(50, int(env("TTS_GROQ_CHAR_LIMIT", "190")))
# canopylabs/orpheus-v1-english di tier gratis: RPM=10, RPD=100
# (https://console.groq.com/docs/rate-limits). Default 9 agar tidak menyentuh
# plafon. RPD=100 artinya SATU video 92 segmen sudah menghabiskan kuota harian,
# karena itu Piper (lokal) dijadikan default.
GROQ_RPM = max(1, int(env("TTS_GROQ_RPM", "9")))

# --- Piper TTS: lokal, offline, tanpa API key & tanpa rate limit ---------
PIPER_VOICE = env("TTS_PIPER_VOICE", "en_US-lessac-medium")
PIPER_DIR = Path(env("TTS_PIPER_DIR", ".piper-voices"))
PIPER_HF_BASE = env("TTS_PIPER_HF_BASE",
                    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0")
# length_scale > 1 = lebih lambat, < 1 = lebih cepat. Kosong = default model.
PIPER_LENGTH_SCALE = env("TTS_PIPER_LENGTH_SCALE")

# Di-set True begitu Edge TTS balas 403 -> sisa segmen pindah provider lain.
_EDGE_BLOCKED = False
_GROQ_DAILY_EXHAUSTED = False   # RPD habis: retry tidak ada gunanya
_ACTIVE_PROVIDER = "?"
_PIPER_VOICE = None             # model Piper di-load sekali per proses
_PIPER_LOCK = threading.Lock()

llm = None  # di-init di main()


# ----------------------------------------------------------------- utils
def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"Command gagal: {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    return p.stdout


class FFmpegError(RuntimeError):
    """Kegagalan ffmpeg yang dilempar sebagai exception biasa.

    Wajib dipakai di kode yang jalan dalam thread/async (to_wav, _concat_wav).
    `sys.exit()` di dalam thread hanya melempar SystemExit yang tidak bisa
    ditangkap `except Exception`, sehingga lolos dari logika retry/fallback TTS.
    """


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


# ----------------------------------------------------------------- 1. STT
def extract_audio():
    """Return list[(path, offset_detik)]. Dipecah per 20 menit jika > 24MB (limit Groq)."""
    full = WORK / "audio.mp3"
    run(["ffmpeg", "-y", "-i", VIDEO, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "48k", full])
    if full.stat().st_size <= 24 * 1024 * 1024:
        return [(full, 0.0)]

    run(["ffmpeg", "-y", "-i", full, "-f", "segment", "-segment_time", "1200",
         "-reset_timestamps", "1", "-c:a", "libmp3lame", "-b:a", "48k",
         WORK / "chunk_%03d.mp3"])
    chunks, offset, res = sorted(WORK.glob("chunk_*.mp3")), 0.0, []
    for c in chunks:
        res.append((c, offset))
        offset += probe_duration(c)
    return res


def groq_transcribe(path: Path) -> dict:
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    r = None
    for attempt in range(5):
        with open(path, "rb") as f:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                files={"file": (path.name, f)},
                data={
                    "model": STT_MODEL,
                    "language": "id",
                    "response_format": "verbose_json",
                    "temperature": "0",
                    "timestamp_granularities[]": "segment",
                },
                timeout=600,
            )
        if r.status_code == 429 or r.status_code >= 500:
            wait = 5 * (2 ** attempt)
            print(f"[groq] {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


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
    segs = []
    for path, offset in extract_audio():
        print(f"[stt] {path.name} (offset {offset:.1f}s)")
        data = groq_transcribe(path)
        for s in data.get("segments", []):
            text = s["text"].strip()
            if text:
                segs.append({"start": s["start"] + offset,
                             "end": s["end"] + offset,
                             "src": text})
    segs.sort(key=lambda x: x["start"])
    return merge_short(segs)


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


# ----------------------------------------------------------------- 3. TTS + fit
TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse")


def to_wav(src: Path, dst: Path, tempo: float = 1.0):
    """audio (mp3/wav) -> wav mono 24k, trim silence di ujung, opsional percepat (atempo).

    Dipanggil dari asyncio.to_thread, jadi pakai run_cmd (raise) bukan run (sys.exit):
    SystemExit di dalam thread tidak tertangkap `except Exception`, sehingga akan
    menembus logika retry/fallback TTS.

    Ekstensi src sengaja generik (.audio): ffmpeg mendeteksi format dari ISI file,
    jadi mp3 dari Edge TTS maupun wav dari Groq sama-sama terbaca.
    """
    f = TRIM + (f",atempo={tempo:.4f}" if tempo > 1.0 else "")
    run_cmd(["ffmpeg", "-y", "-i", src, "-af", f, "-ar", SR, "-ac", 1,
             "-c:a", "pcm_s16le", dst])
    if wav_duration(dst) < 0.05:  # fallback bila trim menghabiskan semua audio
        f = f"atempo={tempo:.4f}" if tempo > 1.0 else "anull"
        run_cmd(["ffmpeg", "-y", "-i", src, "-af", f, "-ar", SR, "-ac", 1,
                 "-c:a", "pcm_s16le", dst])


class TTSBlocked(RuntimeError):
    """Provider menolak secara permanen -- jangan di-retry.

    Contoh: Edge TTS 403 (IP datacenter diblok Microsoft), Groq 401 (key salah),
    atau kuota harian Groq (RPD) sudah habis.
    """


class GroqRateLimited(RuntimeError):
    """429 dari Groq, disertai saran durasi tunggu dalam detik."""

    def __init__(self, msg: str, wait: float = 0.0):
        super().__init__(msg)
        self.wait = wait


class RateLimiter:
    """Jaga jarak antar-request supaya tidak menembus RPM provider.

    Groq Orpheus tier gratis cuma 10 RPM. Tanpa pembatas ini, 92 segmen
    ditembakkan secepat mungkin dan hasilnya 42 dari 92 segmen kena 429
    (run 38286526459). Thread-safe karena _groq_once jalan di to_thread.
    """

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.interval = 60.0 / self.rpm
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if wait > 0:
            time.sleep(wait)


_GROQ_LIMITER = RateLimiter(GROQ_RPM)

# Body error API dipotong sepanjang ini. Dulu 200 karakter, yang membuat URL
# "terms acceptance" dari Groq terpotong sehingga tidak bisa ditindaklanjuti.
ERR_BODY_LIMIT = 600


def _err_body(r) -> str:
    """Ambil error.message kalau responsnya JSON, kalau tidak potong body mentah."""
    if "json" in r.headers.get("content-type", ""):
        try:
            return str(r.json()["error"]["message"])[:ERR_BODY_LIMIT]
        except Exception:  # noqa: BLE001
            pass
    return r.text[:ERR_BODY_LIMIT]


def _retry_after(r, default: float = 20.0) -> float:
    """Header retry-after (detik) yang dikirim Groq saat 429."""
    try:
        return min(max(float(r.headers.get("retry-after", "")), 1.0), 120.0)
    except (TypeError, ValueError):
        return default


def _is_blocked(err: BaseException) -> bool:
    s = f"{type(err).__name__}: {err}"
    return ("403" in s or "Invalid response status" in s
            or "WSServerHandshakeError" in s or "Forbidden" in s)


# --- provider: Edge TTS --------------------------------------------------
async def _edge_once(text: str, out: Path):
    import edge_tts  # lazy: tidak dibutuhkan saat TTS_PROVIDER=groq
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))


# --- provider: Groq Orpheus ---------------------------------------------
def split_chunks(text: str, limit: int = GROQ_CHAR_LIMIT):
    """Pecah teks jadi <= `limit` karakter, utamakan batas kalimat lalu spasi.

    Orpheus menolak input > 200 karakter, jadi segmen panjang wajib dipecah.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text] if text else []
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        while len(sent) > limit:            # satu kalimat pun kepanjangan -> potong paksa
            cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if not sent:
            continue
        if cur and len(cur) + 1 + len(sent) > limit:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out or [text[:limit]]


def _concat_wav(parts, out: Path):
    """Gabung beberapa wav jadi satu.

    `-f wav` WAJIB eksplisit: `out` berekstensi .audio dan ffmpeg menolak menebak
    muxer dari ekstensi tak dikenal ("Unable to choose an output format").
    Re-encode ke 24k mono sekaligus menormalkan parameter antar-chunk.
    """
    lst = out.parent / (out.stem + ".concat.txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    try:
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                 "-f", "wav", "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", out])
    finally:
        lst.unlink(missing_ok=True)


def _groq_once(text: str, out: Path):
    """POST ke Groq audio/speech. Output WAV; kalau teks dipecah, hasil di-concat."""
    global _GROQ_DAILY_EXHAUSTED
    if not GROQ_KEY:
        raise TTSBlocked("GROQ_API_KEY kosong, tidak bisa fallback ke Groq TTS")
    if _GROQ_DAILY_EXHAUSTED:
        raise TTSBlocked("Kuota harian Groq (RPD) sudah habis -- lihat pesan sebelumnya")
    chunks = split_chunks(text) or [""]
    parts = []
    try:
        for n, ch in enumerate(chunks):
            _GROQ_LIMITER.acquire()   # jaga jarak request, hormati RPM
            r = requests.post(
                GROQ_TTS_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}",
                         "Content-Type": "application/json"},
                json={"model": GROQ_TTS_MODEL, "voice": GROQ_TTS_VOICE,
                      "input": ch, "response_format": "wav"},
                timeout=180,
            )
            body = _err_body(r)
            if r.status_code in (401, 403):
                raise TTSBlocked(f"Groq TTS {r.status_code}: {body}")
            if r.status_code == 429:
                # x-ratelimit-remaining-requests = sisa kuota HARIAN (RPD).
                # Kalau 0, retry sampai kapan pun tidak akan berhasil.
                if r.headers.get("x-ratelimit-remaining-requests") == "0":
                    _GROQ_DAILY_EXHAUSTED = True
                    raise TTSBlocked(
                        "Kuota harian Groq (RPD=100) habis. Orpheus tier gratis "
                        "tidak cukup untuk video panjang: 92 segmen = 92 request. "
                        "Pakai TTS_PROVIDER=piper (lokal, tanpa kuota), upgrade "
                        "plan Groq, atau tunggu reset tengah malam.")
                wait = _retry_after(r)
                raise GroqRateLimited(f"Groq TTS 429: {body} (tunggu {wait:.0f}s)", wait)
            if r.status_code >= 500:
                raise RuntimeError(f"Groq TTS {r.status_code}: {body}")
            r.raise_for_status()
            p = out.parent / f"{out.stem}.part{n}.wav"
            p.write_bytes(r.content)
            parts.append(p)
        if not parts:
            raise RuntimeError("Groq TTS tidak menghasilkan audio")
        if len(parts) == 1:
            parts[0].replace(out)
            parts = []
        else:
            _concat_wav(parts, out)
    finally:
        for p in parts:
            p.unlink(missing_ok=True)


# --- provider: Piper TTS (lokal, offline, tanpa kuota) -------------------
def _piper_ensure_model(name: str) -> Path:
    """Pastikan model suara Piper tersedia; download dari HuggingFace kalau belum.

    `name` boleh "en_US-lessac-medium" (di-download ke PIPER_DIR) atau path
    langsung ke file .onnx. File konfigurasi .onnx.json di-load otomatis oleh
    PiperVoice.load() dari samping file model.
    """
    if name.endswith(".onnx"):
        direct = Path(name)
        if not direct.is_file():
            raise TTSBlocked(f"Model Piper tidak ditemukan: {direct}")
        return direct

    parts = name.split("-")
    if len(parts) != 3:
        raise TTSBlocked(
            f"TTS_PIPER_VOICE '{name}' tidak valid. Format: <lang>_<US>-<nama>-"
            f"<kualitas>, contoh en_US-lessac-medium")
    lang, vname, quality = parts
    family = lang.split("_")[0]
    base = f"{PIPER_HF_BASE}/{family}/{lang}/{vname}/{quality}/{name}"

    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    onnx = PIPER_DIR / f"{name}.onnx"
    for dst, url in ((PIPER_DIR / f"{name}.onnx.json", base + ".onnx.json"),
                     (onnx, base + ".onnx")):
        if dst.exists() and dst.stat().st_size > 0:
            continue
        tmp = dst.parent / (dst.name + ".part")
        for attempt in range(3):
            try:
                print(f"[piper] download {url}")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(dst)
                break
            except Exception as e:  # noqa: BLE001
                tmp.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError(
                        f"gagal download voice Piper '{name}' dari {url}: {e}") from e
                time.sleep(3 * (attempt + 1))
    return onnx


def _piper_get_voice():
    """Load model Piper sekali per proses. Thread-safe: dipanggil dari to_thread."""
    global _PIPER_VOICE
    if _PIPER_VOICE is None:
        with _PIPER_LOCK:
            if _PIPER_VOICE is None:          # double-checked locking
                from piper import PiperVoice   # lazy: onnxruntime berat
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
            print(f"[piper] TTS_PIPER_LENGTH_SCALE '{PIPER_LENGTH_SCALE}' bukan angka, diabaikan")
    tmp = out.parent / (out.stem + ".piper.part.wav")
    with wave.open(str(tmp), "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=cfg)
    if not tmp.exists() or tmp.stat().st_size < 100:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Piper menghasilkan audio kosong")
    tmp.replace(out)


# --- dispatcher ----------------------------------------------------------
def _provider_chain():
    """Urutan provider yang boleh dicoba untuk segmen ini."""
    if TTS_PROVIDER in ("edge", "groq", "piper"):
        return [TTS_PROVIDER]
    # auto: Edge (kualitas terbaik, gratis) -> Piper (lokal, selalu jalan)
    #       -> Groq (hanya kalau ada key dan kuota harian belum habis)
    chain = []
    if not _EDGE_BLOCKED:
        chain.append("edge")
    chain.append("piper")
    if GROQ_KEY and not _GROQ_DAILY_EXHAUSTED:
        chain.append("groq")
    return chain


# Jeda antar percobaan (detik) per provider. Groq dibatasi per MENIT, jadi
# backoff pendek ala Edge (2/4/6s) tidak akan pernah berhasil di sana.
BACKOFF = {"edge": 2.0, "groq": 20.0, "piper": 1.0}


async def tts(text: str, out: Path, sem: asyncio.Semaphore) -> str:
    """Generate TTS ke `out`. Return nama provider yang berhasil."""
    global _EDGE_BLOCKED, _ACTIVE_PROVIDER
    async with sem:
        chain = _provider_chain()
        last = None
        for prov in chain:
            tries = {"edge": EDGE_TRIES, "groq": GROQ_TRIES,
                     "piper": PIPER_TRIES}[prov]
            for attempt in range(tries):
                try:
                    if prov == "edge":
                        coro = _edge_once(text, out)
                    elif prov == "groq":
                        coro = asyncio.to_thread(_groq_once, text, out)
                    else:
                        coro = asyncio.to_thread(_piper_once, text, out)
                    # wait_for: koneksi yang menggantung tidak boleh memblokir selamanya
                    await asyncio.wait_for(coro, timeout=TTS_TIMEOUT)
                    if out.exists() and out.stat().st_size > 0:
                        if prov != _ACTIVE_PROVIDER:
                            if _ACTIVE_PROVIDER != "?":
                                print(f"[tts] provider aktif -> {prov}")
                            _ACTIVE_PROVIDER = prov
                        return prov
                    last = RuntimeError("hasil TTS kosong")
                except asyncio.TimeoutError:
                    last = RuntimeError(f"timeout >{TTS_TIMEOUT:.0f}s")
                except GroqRateLimited as e:
                    # 429 sementara: tunggu sesuai retry-after dari server.
                    last = e
                    await asyncio.sleep(e.wait)
                    continue
                except TTSBlocked as e:
                    # Permanen untuk provider ini -> jangan retry, pindah saja.
                    last = e
                    if prov == "edge" and not _EDGE_BLOCKED:
                        _EDGE_BLOCKED = True
                        sisa = chain[chain.index(prov) + 1:] or ["(tidak ada)"]
                        print(f"[tts] Edge TTS diblok (403 / IP datacenter) "
                              f"-> pindah permanen ke: {sisa[0]}")
                    elif prov == "groq":
                        print(f"[tts] Groq tidak bisa dipakai: {str(e)[:200]}")
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    if prov == "edge" and _is_blocked(e):
                        if not _EDGE_BLOCKED:
                            _EDGE_BLOCKED = True
                            sisa = chain[chain.index(prov) + 1:] or ["(tidak ada)"]
                            print(f"[tts] Edge TTS diblok (403 / IP datacenter) "
                                  f"-> pindah permanen ke: {sisa[0]}")
                        break
                if attempt < tries - 1:
                    await asyncio.sleep(min(BACKOFF[prov] * (attempt + 1), 60.0))
        raise RuntimeError(f"TTS gagal untuk: {text!r} via {chain} ({last})")


async def preflight_tts():
    """Tes provider SEBELUM menghabiskan kuota Groq STT + LLM translate.

    Tanpa ini, run baru ketahuan TTS-nya mati setelah 3-5 menit dan setelah
    40-92 segmen Whisper + terjemahan terbuang.
    """
    global _ACTIVE_PROVIDER
    probe = WORK / "_preflight.audio"
    sem = asyncio.Semaphore(1)
    t0 = time.time()
    try:
        prov = await asyncio.wait_for(
            tts("Preflight check for the dubbing pipeline.", probe, sem),
            timeout=TTS_TIMEOUT * 2 + 30)
        print(f"[preflight] TTS OK via {prov} dalam {time.time() - t0:.1f}s")
        _ACTIVE_PROVIDER = prov
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[preflight] Semua provider TTS gagal ({_provider_chain()}): {e}\n"
                 "Edge TTS 403 = IP runner diblok Microsoft. Set TTS_PROVIDER=groq "
                 "dan pastikan GROQ_API_KEY terisi, atau pakai proxy/self-hosted runner.")
    finally:
        probe.unlink(missing_ok=True)


async def build_clip(i: int, seg: dict, sem: asyncio.Semaphore):
    aud = WORK / f"tts_{i:04d}.audio"   # mp3 (edge) atau wav (groq); ffmpeg sniff isi
    wav = WORK / f"clip_{i:04d}.wav"
    tag = WORK / f"tts_{i:04d}.txt"     # isi = teks yang dipakai -> invalidasi cache
    slot = seg["slot"]
    text = seg["en"]

    def cached(t: str) -> bool:
        return (aud.exists() and aud.stat().st_size > 0
                and tag.exists() and tag.read_text(encoding="utf-8") == t)

    if cached(text):
        prov = "cache"
    else:
        prov = await tts(text, aud, sem)
        tag.write_text(text, encoding="utf-8")
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
                tag.write_text(text, encoding="utf-8")
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
        print(f"\n[tts] GAGAL {len(failed)}/{len(segs)} segmen. "
              f"Provider terakhir: {_ACTIVE_PROVIDER}")
        for i, r in failed[:8]:
            print(f"  #{i:04d}: {type(r).__name__}: {str(r)[:180]}")
        if len(failed) > 8:
            print(f"  ... dan {len(failed) - 8} lainnya")
        print("  detail lengkap: output/tts_failures.json")
        if _EDGE_BLOCKED and not GROQ_KEY:
            print("  HINT: Edge TTS kena 403 (IP runner diblok) dan GROQ_API_KEY kosong. "
                  "Set secret GROQ_API_KEY agar fallback aktif.")
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
    if (st.get("video_sha") == video_sha
            and st.get("stt_model") == STT_MODEL
            and st.get("llm_model") == LLM_MODEL
            and isinstance(st.get("segments"), list) and st["segments"]
            and all(s.get("en") for s in st["segments"])):
        return st["segments"]
    print("[cache] state.json tidak cocok (video/model beda) -> hitung ulang")
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
    if not GROQ_KEY:
        sys.exit("GROQ_API_KEY belum di-set")
    if not LLM_KEY:
        sys.exit("LLM_API_KEY belum di-set")
    llm = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE or None)

    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    video_dur = probe_duration(VIDEO)
    print(f"[init] video {video_dur:.1f}s, provider={TTS_PROVIDER}, "
          f"max_speedup={MAX_SPEEDUP}")
    print(f"[init] piper_voice={PIPER_VOICE}, edge_voice={VOICE}, "
          f"groq_voice={GROQ_TTS_VOICE} (model={GROQ_TTS_MODEL}, rpm={GROQ_RPM})")

    video_sha = sha256_file(VIDEO)
    print(f"[init] sha256 video {video_sha[:16]}...")

    # 0) Cek TTS dulu: murah (~2 detik), hemat kuota Groq+LLM kalau ternyata mati.
    asyncio.run(preflight_tts())

    # 1-2) STT + translate (di-cache antar run untuk video yang sama)
    segs = load_state(video_sha)
    if segs is not None:
        print(f"[cache] pakai ulang hasil STT+translate: {len(segs)} segmen "
              "(skip Groq Whisper & LLM)")
        compute_slots(segs, video_dur)
    else:
        segs = transcribe()
        if not segs:
            sys.exit("Tidak ada ucapan terdeteksi")
        print(f"[stt] {len(segs)} segmen")
        compute_slots(segs, video_dur)
        translate(segs)
        save_state(video_sha, segs)

    # 3) TTS
    clips = asyncio.run(build_all(segs))

    # 4) assemble + mux
    dub_wav, dub_dur, placed = assemble(segs, clips, video_dur)
    print(f"[assemble] audio dubbing {dub_dur:.1f}s vs video {video_dur:.1f}s")

    srt = write_srt(placed)
    mux(dub_wav, dub_dur, video_dur, srt if BURN_SUBS else None)
    save_state(video_sha, segs)  # teks final (mungkin sudah di-shorten)
    (OUT / "segments.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] output/dubbed.mp4, dubbed.en.srt, dub.wav, segments.json "
          f"(TTS provider: {_ACTIVE_PROVIDER})")


if __name__ == "__main__":
    main()
