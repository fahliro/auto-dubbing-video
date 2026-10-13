#!/usr/bin/env python3
"""Preflight: pastikan Piper TTS dan model faster-whisper siap SEBELUM pipeline jalan.

Exit 0 = keduanya siap.
Exit 1 = ada yang mati (workflow berhenti di sini, tidak ada waktu STT / kuota
         LLM yang terbuang).

Yang diverifikasi:
  1. Piper TTS  -> voice ter-download, model ter-load, sintesis menghasilkan audio
  2. Whisper    -> model ter-download, ter-load, dan bisa mentranskrip 1 detik
                   audio sunyi (menangkap OOM / nama model salah / library rusak)

Keduanya dipakai lewat kode produksi di dub.py, jadi yang diuji adalah jalur
yang benar-benar akan dijalankan -- bukan tiruan.
"""
import os
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def env(n, d=""):
    return os.environ.get(n, "").strip() or d


def hr(title):
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def make_silence(path: Path, seconds: float = 1.0, rate: int = 16000):
    """WAV sunyi 16 kHz mono, dibuat murni dengan Python (tanpa ffmpeg)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))


# ----------------------------------------------------------------- Piper TTS
def check_piper(voice: str) -> bool:
    hr("PIPER TTS (lokal)")
    import dub
    dub.PIPER_VOICE = voice
    dub.PIPER_DIR = Path(env("TTS_PIPER_DIR", ".piper-voices"))
    dub._PIPER_VOICE = None

    out = Path("work/_preflight_piper.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        dub._piper_once("Preflight check for the dubbing pipeline.", out)
        dt = time.time() - t0
        size = out.stat().st_size if out.exists() else 0
        if size < 1000:
            print(f"[piper] GAGAL: audio terlalu kecil ({size} bytes)")
            return False
        with wave.open(str(out)) as r:
            dur = r.getnframes() / r.getframerate()
            print(f"[piper] OK  {dt:.1f}s  {size} bytes  {dur:.2f}s audio "
                  f"{r.getframerate()} Hz mono")
            print(f"[piper] RTF {dt / max(dur, 0.01):.3f}x "
                  f"({dur / max(dt, 0.001):.1f}x lebih cepat dari realtime)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[piper] GAGAL: {type(e).__name__}: {e}")
        print("  - Cek nama voice: format <lang>_<REGION>-<nama>-<kualitas>,")
        print("    contoh en_US-lessac-medium. Harus ada di rhasspy/piper-voices.")
        print("  - Cek akses runner ke huggingface.co.")
        print("  - Kalau proses di-kill tanpa pesan: onnxruntime kehabisan memori.")
        return False
    finally:
        out.unlink(missing_ok=True)


# ----------------------------------------------------------------- Whisper
def check_whisper(model: str) -> bool:
    hr(f"FASTER-WHISPER (lokal, model={model})")
    import dub
    dub.STT_MODEL = model
    dub.STT_DEVICE = env("STT_DEVICE", "cpu")
    dub.STT_COMPUTE_TYPE = env("STT_COMPUTE_TYPE", "int8")
    dub.STT_THREADS = max(0, int(env("STT_THREADS", "0")))
    dub.STT_MODEL_DIR = Path(env("STT_MODEL_DIR", ".whisper-models"))
    dub._STT = None

    try:
        t0 = time.time()
        m = dub.load_stt_model()          # download (bila perlu) + load
        dt_load = time.time() - t0
        print(f"[whisper] model ter-load dalam {dt_load:.1f}s")

        # transkrip 1 detik sunyi: membuktikan inference benar-benar jalan
        silent = Path("work/_preflight_silence.wav")
        make_silence(silent, seconds=1.0)
        t0 = time.time()
        segs, info = m.transcribe(str(silent), language="id", task="transcribe",
                                  beam_size=1, vad_filter=False)
        got = list(segs)
        dt_run = time.time() - t0
        silent.unlink(missing_ok=True)
        print(f"[whisper] inference OK {dt_run:.1f}s "
              f"({len(got)} segmen dari audio sunyi -- wajar 0/sedikit)")
        print(f"[whisper] compute_type={dub.STT_COMPUTE_TYPE} "
              f"device={dub.STT_DEVICE} threads={dub.STT_THREADS or 'auto'}")

        total = dt_load + dt_run
        print(f"[whisper] total preflight {total:.1f}s")
        if total > 120:
            print("[whisper] catatan: load lama, biasanya karena download model")
            print("           pertama kali. Pastikan step 'Cache model Whisper'")
            print("           aktif supaya run berikutnya tidak download ulang.")
        return True
    except SystemExit as e:
        # load_stt_model() memang sys.exit dengan pesan yang sudah actionable
        print(f"[whisper] GAGAL: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[whisper] GAGAL: {type(e).__name__}: {e}")
        print("  - Nama model valid: tiny, base, small, medium, large-v3,")
        print("    large-v3-turbo, distil-large-v3.")
        print("  - Kalau 'Killed' / tidak ada pesan: kehabisan RAM. Runner")
        print("    GitHub punya 16 GB; coba STT_COMPUTE_TYPE=int8 dan model")
        print("    lebih kecil (small/medium).")
        return False


def main():
    voice = env("TTS_PIPER_VOICE", "en_US-lessac-medium")
    model = env("STT_MODEL", "large-v3-turbo")
    print(f"[preflight] TTS_PIPER_VOICE={voice}  STT_MODEL={model}")
    print(f"[preflight] LLM_BASE_URL={'(diset)' if env('LLM_BASE_URL') else '(default OpenAI)'} "
          f"LLM_MODEL={env('LLM_MODEL', 'gpt-4o-mini')} "
          f"LLM_API_KEY={'ada' if env('LLM_API_KEY') else 'TIDAK ADA'}")

    ok_piper = check_piper(voice)
    ok_whisper = check_whisper(model)

    hr("RINGKASAN")
    print(f"  Piper TTS      : {'OK' if ok_piper else 'GAGAL'}")
    print(f"  faster-whisper : {'OK' if ok_whisper else 'GAGAL'}")
    print(f"  LLM_API_KEY    : {'ada' if env('LLM_API_KEY') else 'TIDAK ADA (wajib)'}")

    if not (ok_piper and ok_whisper and env("LLM_API_KEY")):
        print("::error::Preflight gagal. Lihat detail di atas.")
        return 1
    print("::notice::Preflight OK: Piper TTS + faster-whisper siap, semua lokal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
