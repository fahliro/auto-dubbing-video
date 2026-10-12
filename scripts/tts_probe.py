#!/usr/bin/env python3
"""Probe provider TTS sebelum pipeline mahal (Groq Whisper + LLM) dijalankan.

Exit 0 = minimal satu provider hidup.
Exit 1 = semua mati (workflow berhenti di sini, kuota STT/LLM tidak terpakai).

Dipanggil sebagai step terpisah di workflow supaya kegagalannya kelihatan jelas
di UI Actions, bukan terkubur di tengah log step "Jalankan auto dubbing".
"""
import asyncio
import os
import sys
import time

import requests

PROBE_TEXT = "Preflight check for the dubbing pipeline."
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"


def env(n, d=""):
    return os.environ.get(n, "").strip() or d


async def probe_edge(voice: str, timeout: float = 45.0):
    import edge_tts
    dst = "work/_probe_edge.mp3"
    os.makedirs("work", exist_ok=True)
    t0 = time.time()
    await asyncio.wait_for(
        edge_tts.Communicate(PROBE_TEXT, voice).save(dst), timeout=timeout)
    size = os.path.getsize(dst) if os.path.exists(dst) else 0
    if os.path.exists(dst):
        os.remove(dst)
    if size <= 0:
        raise RuntimeError("Edge TTS menghasilkan file kosong")
    return time.time() - t0, size


# Body error Groq dipotong sepanjang ini supaya URL/instruksi di dalamnya
# tidak terpenggal (versi lama memotong di 200 char -> URL terms acceptance hilang).
ERR_BODY_LIMIT = 900

HINTS = [
    ("requires terms acceptance",
     "Model ini butuh penerimaan TERMS di akun Groq Anda (sekali saja).\n"
     "            Buka URL di bawah dengan akun pemilik GROQ_API_KEY, klik Accept,\n"
     "            lalu re-run workflow. Tidak ada yang perlu diubah di repo."),
    ("model_not_found",
     "Model tidak tersedia untuk akun/key ini. Cek nama model di\n"
     "            https://console.groq.com/docs/models dan set TTS_GROQ_MODEL."),
    ("Invalid API Key",
     "Secret GROQ_API_KEY tidak valid / sudah di-revoke. Perbarui di\n"
     "            Settings > Secrets and variables > Actions."),
    ("rate_limit",
     "Groq rate limit. Turunkan tts_concurrency ke 2, atau tunggu sebentar."),
    ("billing",
     "Masalah billing/kuota di akun Groq."),
]


def hint_for(body: str) -> str:
    low = body.lower()
    for needle, msg in HINTS:
        if needle.lower() in low:
            return msg
    return ""


def probe_groq(voice: str, model: str, timeout: float = 60.0):
    key = env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY tidak ada")
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
        ctype = r.headers.get("content-type", "")
        body = r.text[:ERR_BODY_LIMIT]
        if "json" in ctype:
            # ambil field message saja kalau bisa, biar lebih enak dibaca
            try:
                body = r.json()["error"]["message"][:ERR_BODY_LIMIT]
            except Exception:  # noqa: BLE001
                pass
        err = RuntimeError(f"HTTP {r.status_code}: {body}")
        err.hint = hint_for(body)          # type: ignore[attr-defined]
        raise err
    if len(r.content) < 1000:
        raise RuntimeError(f"audio terlalu kecil ({len(r.content)} bytes)")
    return time.time() - t0, len(r.content)


def blocked(err) -> bool:
    s = f"{type(err).__name__}: {err}"
    return "403" in s or "Invalid response status" in s or "WSServerHandshakeError" in s


def main():
    provider = env("TTS_PROVIDER", "auto").lower()
    edge_voice = env("TTS_VOICE", "en-US-AndrewNeural")
    groq_voice = env("TTS_GROQ_VOICE", "troy")
    groq_model = env("TTS_GROQ_MODEL", "canopylabs/orpheus-v1-english")
    has_groq = bool(env("GROQ_API_KEY"))

    print(f"[probe] TTS_PROVIDER={provider} edge_voice={edge_voice} "
          f"groq_voice={groq_voice} groq_key={'ada' if has_groq else 'TIDAK ADA'}")

    ok = []

    if provider in ("auto", "edge"):
        try:
            dt, size = asyncio.run(probe_edge(edge_voice))
            print(f"[probe] EDGE  OK   {dt:.1f}s  {size} bytes")
            ok.append("edge")
        except Exception as e:  # noqa: BLE001
            tag = "DIBLOKIR 403 (IP datacenter)" if blocked(e) else "gagal"
            print(f"[probe] EDGE  {tag}: {type(e).__name__}: {str(e)[:200]}")

    if provider in ("auto", "groq"):
        if not has_groq:
            print("[probe] GROQ  dilewati (GROQ_API_KEY kosong)")
        else:
            try:
                dt, size = probe_groq(groq_voice, groq_model)
                print(f"[probe] GROQ  OK   {dt:.1f}s  {size} bytes")
                ok.append("groq")
            except Exception as e:  # noqa: BLE001
                print(f"[probe] GROQ  gagal: {type(e).__name__}: {e}")
                hint = getattr(e, "hint", "")
                if hint:
                    print("[probe] HINT: " + hint)
                    print("::error title=Groq TTS butuh aksi di console::"
                          + hint.replace("\n", " ").replace("            ", ""))

    if ok:
        print(f"[probe] provider siap pakai: {', '.join(ok)} -> lanjut ke pipeline")
        print("::notice::TTS provider aktif: " + ", ".join(ok))
        return 0

    print("\n[probe] SEMUA provider TTS mati.")
    print("  Kemungkinan penyebab:")
    print("  - GROQ 400 'requires terms acceptance'")
    print("        => terima terms model di https://console.groq.com (sekali saja),")
    print("           lalu re-run. Ini yang paling sering terjadi.")
    print("  - GROQ 401 'Invalid API Key' => perbarui secret GROQ_API_KEY.")
    print("  - EDGE 403 'Invalid response status'")
    print("        => Microsoft memblokir IP datacenter runner GitHub Actions.")
    print("           Di luar kendali repo; re-run biasanya tetap gagal.")
    print("           Solusi: proxy Edge TTS lewat Cloudflare Worker, atau")
    print("           self-hosted runner dengan IP non-datacenter.")
    print("::error::Semua provider TTS gagal. Lihat log step 'Cek provider TTS'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
