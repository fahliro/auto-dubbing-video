#!/usr/bin/env python3
"""Preflight: pastikan TTS lokal dan model faster-whisper siap SEBELUM pipeline jalan.

Exit 0 = semuanya siap.
Exit 1 = ada yang mati (workflow berhenti di sini, tidak ada waktu STT / kuota
         LLM yang terbuang).

Yang diverifikasi:
  1. TTS -> voice/model ter-download, ter-load, dan sintesis menghasilkan audio
  2. STT -> sesuai STT_PROVIDER:
      groq  : API key valid, model bisa diakses, kuota harian belum habis
              (1 request uji dengan audio sunyi 1 detik)
      local : model faster-whisper ter-download, ter-load, dan bisa mentranskrip
              1 detik audio sunyi (menangkap OOM / nama model salah / lib rusak)

Provider TTS mengikuti TTS_PROVIDER:
  piper   -> default. RTF ~0.23x di 2 vCPU, voice ~63 MB.
  kokoro  -> opsi. Prosodi lebih natural tapi ~6x lebih lambat (RTF ~1.4x di
             2 vCPU). Model int8 92 MB + voices 28 MB, 54 voice tersedia.
Keduanya lokal: tanpa API key, tanpa kuota, tanpa rate limit, tak diblok per IP.

Semua pemeriksaan memanggil kode produksi di dub.py, jadi yang diuji adalah
jalur yang benar-benar akan dijalankan -- bukan tiruan.
"""
import os
import struct
import sys
import time
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_TEXT = "Preflight check for the dubbing pipeline."
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Body error dipotong sepanjang ini. Dulu 200 karakter, yang pernah membuat
# URL "terms acceptance" dari sebuah API terpotong sehingga tidak bisa
# ditindaklanjuti dari log.
ERR_BODY_LIMIT = 900


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


# ---------------------------------------------------- TTS lokal (Piper/Kokoro)
def check_tts(provider: str) -> bool:
    import dub
    dub.TTS_PROVIDER = provider
    dub.PIPER_VOICE = env("TTS_PIPER_VOICE", "en_US-lessac-medium")
    dub.PIPER_DIR = Path(env("TTS_PIPER_DIR", ".piper-voices"))
    dub.KOKORO_VOICE = env("TTS_KOKORO_VOICE", "af_bella")
    dub.KOKORO_DIR = Path(env("TTS_KOKORO_DIR", ".kokoro-models"))
    dub.KOKORO_LANG = env("TTS_KOKORO_LANG", "en-us")
    dub.KOKORO_SPEED = float(env("TTS_KOKORO_SPEED", "1.0") or 1.0)
    dub.KOKORO_MODEL_FILE = env("TTS_KOKORO_MODEL_FILE", "kokoro-v1.0.int8.onnx")
    dub.KOKORO_VOICES_FILE = env("TTS_KOKORO_VOICES_FILE", "voices-v1.0.bin")
    dub._PIPER_VOICE = None
    dub._KOKORO = None

    hr(f"{provider.upper()} TTS (lokal)")
    out = Path("work/_preflight_tts.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        dub._tts_impl()(PROBE_TEXT, out)
        dt = time.time() - t0
        size = out.stat().st_size if out.exists() else 0
        if size < 1000:
            print(f"[{provider}] GAGAL: audio terlalu kecil ({size} bytes)")
            return False
        with wave.open(str(out)) as r:
            dur = r.getnframes() / r.getframerate()
            sr = r.getframerate()
        rtf = dt / max(dur, 0.01)
        print(f"[{provider}] OK  {dt:.1f}s  {size} bytes  {dur:.2f}s audio {sr} Hz mono")
        print(f"[{provider}] RTF {rtf:.3f}x "
              f"({max(dur, 0.01) / max(dt, 0.001):.1f}x lebih cepat dari realtime)")
        # Probe ini cuma ~2 detik audio, jadi biaya tetap (load model, setup
        # fonemizer) mendominasi dan RTF-nya jauh lebih buruk dari kenyataan.
        # Angka produksi yang terukur di runner 2 vCPU, pada 130 segmen nyata:
        #   piper  ~0.23x   kokoro ~1.4x
        print(f"[{provider}] catatan: RTF di atas dari probe {dur:.1f}s sehingga")
        print("           overhead load/setup masih dominan -- tidak representatif.")
        if provider == "kokoro":
            print("           RTF produksi Kokoro ~1.4x (lebih lambat dari realtime),")
            print("           ~6x lebih lambat dari piper (~0.23x). Untuk video")
            print("           panjang selisihnya beberapa menit; kualitas prosodinya")
            print("           yang jadi alasan memilih Kokoro, bukan kecepatannya.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{provider}] GAGAL: {type(e).__name__}: {e}")
        if provider == "kokoro":
            print("  - Voice harus salah satu dari 54 nama bawaan, mis. af_bella,")
            print("    af_sarah, af_nicole, am_adam, am_michael, bm_george.")
            print("  - Model HARUS varian int8 atau fp32. Varian fp16 (q8f16 /")
            print("    q4f16 / fp16) TERUKUR SEGFAULT di onnxruntime CPU -- itu")
            print("    crash native, bukan exception Python, jadi tak tertangkap.")
            print("  - 'Failed to allocate memory': teks terlalu panjang atau RAM")
            print("    kurang. Turunkan TTS_KOKORO_MAX_CHARS (default 500).")
        else:
            print("  - Cek nama voice: format <lang>_<REGION>-<nama>-<kualitas>,")
            print("    contoh en_US-lessac-medium. Harus ada di rhasspy/piper-voices.")
            print("  - Cek akses runner ke huggingface.co.")
            print("  - Kalau proses mati tanpa pesan: onnxruntime kehabisan memori.")
        return False
    finally:
        out.unlink(missing_ok=True)


# ------------------------------------------------------- STT: Groq Whisper
def check_stt_groq(model: str) -> bool:
    """Kirim 1 detik audio sunyi ke Groq Whisper.

    Biaya: 1 request dari RPD 2000 dan 1 audio-second dari ASD 28800 -- cukup
    murah untuk memvalidasi key, akses model, dan sisa kuota SEBELUM pipeline
    menghabiskan waktu extract audio dan kuota LLM.
    """
    hr(f"GROQ WHISPER (cloud, model={model})")
    key = env("GROQ_API_KEY")
    if not key:
        print("[groq-stt] GAGAL: secret GROQ_API_KEY kosong")
        print("  - Isi di Settings > Secrets and variables > Actions, ATAU")
        print("  - pakai stt_provider=local (faster-whisper di runner, tanpa key).")
        print("::error::GROQ_API_KEY kosong padahal stt_provider=groq.")
        return False

    silent = Path("work/_preflight_silence.wav")
    make_silence(silent, seconds=1.0)
    try:
        t0 = time.time()
        with open(silent, "rb") as f:
            r = requests.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (silent.name, f)},
                data={"model": model, "language": "id",
                      "response_format": "verbose_json", "temperature": "0",
                      "timestamp_granularities[]": "segment"},
                timeout=120)
        dt = time.time() - t0
        sisa_rpd = r.headers.get("x-ratelimit-remaining-requests", "?")
        sisa_asd = r.headers.get("x-ratelimit-remaining-audio-seconds", "?")

        if r.status_code == 200:
            n = len(r.json().get("segments", []) or [])
            print(f"[groq-stt] OK  {dt:.1f}s  ({n} segmen dari audio sunyi -- "
                  "wajar 0/sedikit)")
            print(f"[groq-stt] sisa kuota: RPD {sisa_rpd}, audio-seconds {sisa_asd}")
            if sisa_rpd not in ("?", "") and str(sisa_rpd).isdigit() \
                    and int(sisa_rpd) < 5:
                print(f"[groq-stt] PERINGATAN: sisa kuota harian tinggal {sisa_rpd} "
                      "request. Video panjang yang dipecah bisa kehabisan di tengah.")
            return True

        body = r.text[:ERR_BODY_LIMIT]
        if "json" in r.headers.get("content-type", ""):
            try:
                body = str(r.json()["error"]["message"])[:ERR_BODY_LIMIT]
            except Exception:  # noqa: BLE001
                pass
        print(f"[groq-stt] GAGAL: HTTP {r.status_code}: {body}")
        if r.status_code in (401, 403):
            print("  - GROQ_API_KEY tidak valid / sudah di-revoke / tidak punya")
            print("    akses ke model ini.")
        elif r.status_code == 429:
            print("  - Kuota habis atau rate limit. Tunggu reset, atau pakai")
            print("    stt_provider=local.")
        elif r.status_code == 404:
            print(f"  - Model '{model}' tidak dikenal Groq. Pilihan yang valid:")
            print("    whisper-large-v3, whisper-large-v3-turbo.")
        else:
            print("  - Coba stt_provider=local untuk transkripsi di runner tanpa API.")
        print(f"::error::Groq STT gagal (HTTP {r.status_code}).")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[groq-stt] GAGAL: {type(e).__name__}: {e}")
        print("  - Cek koneksi runner ke api.groq.com.")
        return False
    finally:
        silent.unlink(missing_ok=True)


# ---------------------------------------------------- STT: faster-whisper lokal
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
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        print(f"[whisper] GAGAL: {msg[:ERR_BODY_LIMIT]}")
        if isinstance(e, MemoryError) or "alloc" in low or "killed" in low \
                or "memory" in low:
            print("  - Kehabisan RAM. Runner GitHub repo ini terukur 2 vCPU / 7 GB.")
            print("    Urutan yang menolong: STT_THREADS=1, STT_COMPUTE_TYPE=int8,")
            print("    lalu model lebih kecil (medium -> small -> base).")
            print("    Perkiraan RAM: turbo ~3 GB, medium ~2 GB, small ~1 GB,")
            print("    base ~0.5 GB.")
        elif isinstance(e, TypeError) or "unexpected keyword" in low \
                or "has no attribute" in low or "import" in low:
            print("  - Ini BUKAN soal model atau RAM: ini ketidakcocokan versi")
            print("    dependensi. Tersangka utama PyAV -- faster-whisper 1.2.1")
            print("    memanggil av.open(..., metadata_errors='ignore') yang")
            print("    dihapus di PyAV 19. Pastikan requirements.txt memuat")
            print("    'av>=12,<19' dan step install memakai -r requirements.txt.")
        else:
            print("  - Nama model valid: tiny, base, small, medium, large-v3,")
            print("    large-v3-turbo, distil-large-v3.")
            print("  - Kalau download dari huggingface.co gagal, jalankan ulang")
            print("    (model di-cache antar run).")
        return False


# ----------------------------------------------------------------- versi
def report_versions(provider: str) -> bool:
    """Cetak versi dependensi biner.

    Murah, dan membuat masalah ketidakcocokan versi langsung terbaca di log
    tanpa perlu dibedah. Contoh nyata: sebuah run gagal karena PyAV 19 menghapus
    parameter `metadata_errors` yang dipakai faster-whisper 1.2.1 -- kalau
    versinya tercetak di baris awal, penyebabnya langsung kelihatan.
    """
    hr("VERSI DEPENDENSI")
    ok = True
    stt_local = env("STT_PROVIDER", "groq").lower() == "local"
    # kokoro_onnx hanya wajib kalau TTS provider-nya kokoro. faster_whisper, av,
    # dan ctranslate2 hanya wajib kalau STT-nya lokal -- saat STT_PROVIDER=groq
    # ketiganya tidak dipakai sama sekali, jadi tidak boleh membuat preflight gagal.
    optional = set() if provider == "kokoro" else {"kokoro_onnx"}
    if not stt_local:
        optional |= {"faster_whisper", "av", "ctranslate2"}
    for mod, attr in (("faster_whisper", "__version__"), ("av", "__version__"),
                      ("ctranslate2", "__version__"), ("onnxruntime", "__version__"),
                      ("piper", None), ("kokoro_onnx", None),
                      ("requests", "__version__"), ("openai", "__version__")):
        try:
            m = __import__(mod)
            v = getattr(m, attr, "?") if attr else "terpasang"
            print(f"  {mod:<16} {v}")
        except Exception as e:  # noqa: BLE001
            if mod in optional:
                print(f"  {mod:<16} tidak terpasang (tidak perlu untuk {provider})")
                continue
            print(f"  {mod:<16} GAGAL DIIMPOR: {type(e).__name__}: {e}")
            ok = False

    # PyAV >= 19 tidak kompatibel dengan faster-whisper 1.2.1.
    # Hanya relevan kalau STT lokal yang dipakai.
    try:
        if not stt_local:
            raise ImportError("dilewati: STT_PROVIDER bukan local")
        import av
        if int(str(av.__version__).split(".")[0]) >= 19:
            print(f"  !! PyAV {av.__version__} >= 19 TIDAK kompatibel dengan "
                  "faster-whisper 1.2.1")
            print("     (av.open() kehilangan parameter 'metadata_errors').")
            print("     Pin 'av>=12,<19' di requirements.txt.")
            ok = False
    except Exception:  # noqa: BLE001
        pass

    print(f"  python           {sys.version.split()[0]}")
    print(f"  cpu              {os.cpu_count()} core")
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    print(f"  ram              {int(line.split()[1]) / 1048576:.1f} GB")
                    break
    except Exception:  # noqa: BLE001
        pass
    return ok


# ----------------------------------------------------------------- main
def main():
    provider = env("TTS_PROVIDER", "piper").lower()
    if provider not in ("piper", "kokoro"):
        print(f"[preflight] TTS_PROVIDER '{provider}' tidak dikenal. "
              "Pilihan: piper, kokoro.")
        print("::error::TTS_PROVIDER tidak valid (harus piper atau kokoro).")
        return 1

    voice = (env("TTS_KOKORO_VOICE", "af_bella") if provider == "kokoro"
             else env("TTS_PIPER_VOICE", "en_US-lessac-medium"))
    stt_provider = env("STT_PROVIDER", "groq").lower()
    if stt_provider not in ("groq", "local"):
        print(f"[preflight] STT_PROVIDER '{stt_provider}' tidak dikenal. "
              "Pilihan: groq, local.")
        print("::error::STT_PROVIDER tidak valid (harus groq atau local).")
        return 1
    model = (env("STT_GROQ_MODEL", "whisper-large-v3") if stt_provider == "groq"
             else env("STT_MODEL", "large-v3-turbo"))
    print(f"[preflight] TTS_PROVIDER={provider} voice={voice}")
    print(f"[preflight] STT_PROVIDER={stt_provider} model={model}")
    print(f"[preflight] LLM_BASE_URL={'(diset)' if env('LLM_BASE_URL') else '(default OpenAI)'} "
          f"LLM_MODEL={env('LLM_MODEL', 'gpt-4o-mini')} "
          f"LLM_API_KEY={'ada' if env('LLM_API_KEY') else 'TIDAK ADA'}")

    ok_deps = report_versions(provider)
    ok_tts = check_tts(provider)
    ok_stt = (check_stt_groq(model) if stt_provider == "groq"
              else check_whisper(model))

    hr("RINGKASAN")
    print(f"  Dependensi     : {'OK' if ok_deps else 'TIDAK KOMPATIBEL'}")
    print(f"  {provider.capitalize()} TTS".ljust(17) + f": {'OK' if ok_tts else 'GAGAL'}")
    stt_label = "Groq Whisper" if stt_provider == "groq" else "faster-whisper"
    print(f"  {stt_label}".ljust(17) + f": {'OK' if ok_stt else 'GAGAL'}")
    print(f"  LLM_API_KEY    : {'ada' if env('LLM_API_KEY') else 'TIDAK ADA (wajib)'}")
    if stt_provider == "groq":
        print(f"  GROQ_API_KEY   : {'ada' if env('GROQ_API_KEY') else 'TIDAK ADA (wajib)'}")

    if not (ok_deps and ok_tts and ok_stt and env("LLM_API_KEY")
            and (env("GROQ_API_KEY") if stt_provider == "groq" else True)):
        print("::error::Preflight gagal. Lihat detail di atas.")
        return 1
    print(f"::notice::Preflight OK: TTS {provider} (lokal) + STT {stt_label} siap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
