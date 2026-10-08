#!/usr/bin/env python3
"""Auto dubbing Indonesia -> English.

Pipeline:
  video -> ffmpeg (extract audio) -> Groq Whisper (segments + timestamp)
        -> LLM (translate, dubbing-friendly) -> Edge TTS (per segment)
        -> ffmpeg (trim silence, shorten text / speed up if too long)
        -> assemble timeline -> mux ke video (extend video jika audio masih lebih panjang)
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import edge_tts
import requests
from openai import OpenAI


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
CPS = float(env("CHARS_PER_SEC", "14"))  # estimasi karakter/detik TTS inggris
SHORTEN_RATIO = 1.2  # jika audio > slot * 1.2 -> minta LLM mempersingkat dulu
SR = 24000  # sample rate audio dubbing

llm = None  # di-init di main()


# ----------------------------------------------------------------- utils
def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"Command gagal: {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    return p.stdout


def probe_duration(path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def has_audio(path) -> bool:
    out = run(["ffprobe", "-v", "error", "-select_streams", "a",
               "-show_entries", "stream=index", "-of", "csv=p=0", path])
    return bool(out.strip())


def wav_duration(path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


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
                    and (s["end"] - p["start"]) < max_len):
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
    """mp3 -> wav mono 24k, trim silence di ujung, opsional percepat (atempo)."""
    f = TRIM + (f",atempo={tempo:.4f}" if tempo > 1.0 else "")
    run(["ffmpeg", "-y", "-i", src, "-af", f, "-ar", SR, "-ac", 1,
         "-c:a", "pcm_s16le", dst])
    if wav_duration(dst) < 0.05:  # fallback bila trim menghabiskan semua audio
        f = f"atempo={tempo:.4f}" if tempo > 1.0 else "anull"
        run(["ffmpeg", "-y", "-i", src, "-af", f, "-ar", SR, "-ac", 1,
             "-c:a", "pcm_s16le", dst])


async def tts(text: str, out: Path, sem: asyncio.Semaphore):
    async with sem:
        last = None
        for attempt in range(5):
            try:
                await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))
                if out.exists() and out.stat().st_size > 0:
                    return
            except Exception as e:  # noqa: BLE001
                last = e
            await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Edge TTS gagal untuk: {text!r} ({last})")


async def build_clip(i: int, seg: dict, sem: asyncio.Semaphore):
    mp3 = WORK / f"tts_{i:04d}.mp3"
    wav = WORK / f"clip_{i:04d}.wav"
    slot = seg["slot"]
    text = seg["en"]

    await tts(text, mp3, sem)
    await asyncio.to_thread(to_wav, mp3, wav)
    dur = wav_duration(wav)
    info = "ok"

    # 1) Terlalu panjang -> minta LLM mempersingkat, lalu generate ulang
    if dur > slot * SHORTEN_RATIO:
        target = max(6, int(len(text) * slot * 1.05 / dur))
        new_text = await asyncio.to_thread(shorten, text, target)
        if new_text and new_text != text:
            text = new_text
            await tts(text, mp3, sem)
            await asyncio.to_thread(to_wav, mp3, wav)
            dur = wav_duration(wav)
            info = "shortened"

    # 2) Masih lebih panjang dari slot -> percepat (atempo), dibatasi MAX_SPEEDUP
    tempo = 1.0
    if dur > slot:
        tempo = min(dur / slot, MAX_SPEEDUP)
        if tempo > 1.02:
            await asyncio.to_thread(to_wav, mp3, wav, tempo)
            dur = wav_duration(wav)
            info += f"+x{tempo:.2f}"

    # 3) Jika masih > slot (melewati MAX_SPEEDUP) -> akan mendorong segmen berikutnya
    if dur > slot + 0.05:
        info += " OVERFLOW"

    seg["en"] = text
    print(f"[tts] #{i:04d} slot={slot:.2f}s dur={dur:.2f}s {info}")
    return wav, dur


async def build_all(segs):
    sem = asyncio.Semaphore(4)
    return await asyncio.gather(*(build_clip(i, s, sem) for i, s in enumerate(segs)))


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


def mux(dub_wav: Path, dub_dur: float, video_dur: float):
    out = OUT / "dubbed.mp4"
    extend = dub_dur - video_dur  # audio lebih panjang dari video?
    fc = []

    if extend > 0.3:
        print(f"[mux] audio lebih panjang {extend:.2f}s -> video diperpanjang (freeze frame terakhir)")
        fc.append(f"[0:v]tpad=stop_mode=clone:stop_duration={extend:.3f}[v]")
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


def write_srt(placed):
    lines = []
    for i, p in enumerate(placed, 1):
        lines += [str(i), f"{ts(p['start'])} --> {ts(p['end'])}", p["text"], ""]
    (OUT / "dubbed.en.srt").write_text("\n".join(lines), encoding="utf-8")


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
    print(f"[init] video {video_dur:.1f}s, voice={VOICE}, max_speedup={MAX_SPEEDUP}")

    segs = transcribe()
    if not segs:
        sys.exit("Tidak ada ucapan terdeteksi")
    print(f"[stt] {len(segs)} segmen")

    compute_slots(segs, video_dur)
    translate(segs)

    clips = asyncio.run(build_all(segs))
    dub_wav, dub_dur, placed = assemble(segs, clips, video_dur)
    print(f"[assemble] audio dubbing {dub_dur:.1f}s vs video {video_dur:.1f}s")

    mux(dub_wav, dub_dur, video_dur)
    write_srt(placed)
    (OUT / "segments.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] output/dubbed.mp4, dubbed.en.srt, dub.wav, segments.json")


if __name__ == "__main__":
    main()
