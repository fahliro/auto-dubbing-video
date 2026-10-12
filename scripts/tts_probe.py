#!/usr/bin/env python3
"""Probe provider TTS sebelum pipeline mahal (Groq Whisper + LLM) dijalankan.

Exit 0 = minimal satu provider hidup.
Exit 1 = semua mati (workflow berhenti di sini, kuota STT/LLM tidak terpakai).

Dipanggil sebagai step terpisah di workflow supaya kegagalannya kelihatan jelas
di UI Actions, bukan terkubur di tengah log step "Jalankan auto dubbing".

Provider yang diuji mengikuti TTS_PROVIDER:
  piper  -> sintesis lokal (default; tanpa API key, tanpa rate limit)
  groq   -> Groq Orpheus (perlu GROQ_API_KEY + terms diterima + kuota RPD)
  edge   -> Edge TTS (sering 403 dari IP datacenter GitHub Actions)
  auto   -> ketiganya
"""
import asyncio
import os
import sys
import time
from pathlib import Path

import requests

PROBE_TEXT = "Preflight check for the dubbing pipeline."
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"

# Body error dipotong sepanjang ini. Dulu 200 karakter, yang membuat URL
# "terms acceptance" dari Groq terpotong sehingga tidak bisa ditindaklanjuti.
ERR_BODY_LIMIT = 900

HINTS = [
    ("requires terms acceptance",
     "Model ini butuh penerimaan TERMS di akun Groq Anda (sekali saja).\n"
     "            Buka URL di bawah dengan akun pemilik GROQ_API_KEY, klik Accept,\n"
     "            lalu re-run workflow. Tidak ada yang perlu diubah di repo."),
    ("rate limit reached",
     "Groq rate limit. Orpheus tier gratis hanya RPM=10 dan RPD=100/hari,\n"
     "            tidak cukup untuk video panjang. Pakai tts_provider=piper (lokal,\n"
     "            tanpa kuota) atau upgrade plan Groq."),
    ("model_not_found",
     "Model tidak tersedia untuk akun/key ini. Cek nama model di\n"
     "            https://console.groq.com/docs/models dan set TTS_GROQ_MODEL."),
    ("invalid api key",
     "Secret GROQ_API_KEY tidak valid / sudah di-revoke. Perbarui di\n"
     "            Settings > Secrets and variables > Actions."),
    ("billing",
     "Masalah billing/kuota di akun Groq."),
]


def env(n, d=""):
    return os.environ.get(n, "").strip() or d


def hint_for(body: str) -> str:
    low = body.lower()
    for needle, msg in HINTS:
        if needle in low:
            return msg
    return ""


def _err_body(r) -> str:
    if "json" in r.headers.get("content-type", ""):
        try:
            return str(r.json()["error"]["message"])[:ERR_BODY_LIMIT]
        except Exception:  # noqa: BLE001
            pass
    return r.text[:ERR_BODY_LIMIT]


class ProbeError(RuntimeError):
    def __init__(self, msg, hint=""):
        super().__init__(msg)
        self.hint = hint


# --- Piper (lokal) -------------------------------------------------------
def probe_piper(voice: str):
    """Pakai kode produksi di dub.py supaya yang diuji jalur yang sama persis."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dub
    dub.PIPER_VOICE = voice
    dub._PIPER_VOICE = None
    out = Path("work/_probe_piper.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dub._piper_once(PROBE_TEXT, out)
    dt = time.time() - t0
    size = out.stat().st_size if out.exists() else 0
    out.unlink(missing_ok=True)
    if size < 1000:
        raise ProbeError(f"Piper menghasilkan audio terlalu kecil ({size} bytes)")
    return dt, size


# --- Groq Orpheus --------------------------------------------------------
def probe_groq(voice: str, model: str, timeout: float = 60.0):
    key = env("GROQ_API_KEY")
    if not key:
        raise ProbeError("GROQ_API_KEY tidak ada")
    t0 = time.time()
    r = requests.post(
        GROQ_TTS_URL,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"model": model, "voice": voice,
              "input": PROBE_TEXT, "response_format": "wav"},
        timeout=timeout,
    )
    if r.status_code != 200:
        body = _err_body(r)
        extra = ""
        if r.status_code == 429:
            extra = (f" [sisa kuota harian: "
                     f"{r.headers.get('x-ratelimit-remaining-requests', '?')}, "
                     f"retry-after: {r.headers.get('retry-after', '-')}]")
        raise ProbeError(f"HTTP {r.status_code}: {body}{extra}", hint_for(body))
    if len(r.content) < 1000:
        raise ProbeError(f"audio terlalu kecil ({len(r.content)} bytes)")
    return time.time() - t0, len(r.content)


# --- Edge TTS ------------------------------------------------------------
async def probe_edge(voice: str, timeout: float = 45.0):
    import edge_tts  # lazy
    dst = "work/_probe_edge.mp3"
    os.makedirs("work", exist_ok=True)
    t0 = time.time()
    await asyncio.wait_for(
        edge_tts.Communicate(PROBE_TEXT, voice).save(dst), timeout=timeout)
    size = os.path.getsize(dst) if os.path.exists(dst) else 0
    if os.path.exists(dst):
        os.remove(dst)
    if size <= 0:
        raise ProbeError("Edge TTS menghasilkan file kosong")
    return time.time() - t0, size


def blocked(err) -> bool:
    s = f"{type(err).__name__}: {err}"
    return "403" in s or "Invalid response status" in s or "WSServerHandshakeError" in s


def report(label, fn, *a):
    """Jalankan satu probe, cetak hasil, return True kalau hidup."""
    try:
        dt, size = fn(*a)
        print(f"[probe] {label:<6}OK   {dt:.1f}s  {size} bytes")
        return True
    except Exception as e:  # noqa: BLE001
        tag = "DIBLOKIR 403 (IP datacenter)" if (label == "EDGE" and blocked(e)) else "gagal"
        print(f"[probe] {label:<6}{tag}: {type(e).__name__}: {e}")
        hint = getattr(e, "hint", "") or (hint_for(str(e)) if label == "GROQ" else "")
        if hint:
            print("[probe] HINT: " + hint)
            print("::error title=%s TTS butuh aksi::%s"
                  % (label, hint.replace("\n", " ").replace("            ", " ")))
        return False


def main():
    provider = env("TTS_PROVIDER", "piper").lower()
    edge_voice = env("TTS_VOICE", "en-US-AndrewNeural")
    groq_voice = env("TTS_GROQ_VOICE", "troy")
    groq_model = env("TTS_GROQ_MODEL", "canopylabs/orpheus-v1-english")
    piper_voice = env("TTS_PIPER_VOICE", "en_US-lessac-medium")
    has_groq = bool(env("GROQ_API_KEY"))

    print(f"[probe] TTS_PROVIDER={provider} piper_voice={piper_voice} "
          f"groq_voice={groq_voice} edge_voice={edge_voice} "
          f"groq_key={'ada' if has_groq else 'TIDAK ADA'}")

    ok = []
    if provider in ("auto", "piper"):
        if report("PIPER", probe_piper, piper_voice):
            ok.append("piper")
    if provider in ("auto", "edge"):
        if report("EDGE", lambda v: asyncio.run(probe_edge(v)), edge_voice):
            ok.append("edge")
    if provider in ("auto", "groq"):
        if not has_groq:
            print("[probe] GROQ   dilewati (GROQ_API_KEY kosong)")
        elif report("GROQ", probe_groq, groq_voice, groq_model):
            ok.append("groq")

    if ok:
        print(f"[probe] provider siap pakai: {', '.join(ok)} -> lanjut ke pipeline")
        print("::notice::TTS provider aktif: " + ", ".join(ok))
        return 0

    print("\n[probe] SEMUA provider TTS mati. Kemungkinan penyebab:")
    print("  - PIPER gagal download/load model")
    print("        => cek nama voice (format en_US-lessac-medium) dan akses ke")
    print("           huggingface.co dari runner.")
    print("  - GROQ 400 'requires terms acceptance'")
    print("        => terima terms model di https://console.groq.com (sekali saja).")
    print("  - GROQ 429 / RPD habis")
    print("        => Orpheus tier gratis cuma 100 request/HARI. Video 92 segmen")
    print("           sudah menghabisinya. Pakai piper, atau upgrade plan.")
    print("  - GROQ 401 'Invalid API Key' => perbarui secret GROQ_API_KEY.")
    print("  - EDGE 403 'Invalid response status'")
    print("        => Microsoft memblokir IP datacenter runner GitHub Actions.")
    print("           Di luar kendali repo; re-run biasanya tetap gagal.")
    print("::error::Semua provider TTS gagal. Lihat log step 'Cek provider TTS'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
