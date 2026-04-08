#!/usr/bin/env python3
"""
RunPod Serverless Handler — Confirmili STT + TTS
Receives audio or text, returns transcription or synthesized audio.

Input for STT:
    {"action": "stt", "audio": "<base64 wav>"}

Input for TTS:
    {"action": "tts", "text": "...", "emotion": "professional", "voice": "aisha_happy"}

Output:
    {"text": "..."} or {"audio": "<base64 wav>"}
"""
import os, sys, io, base64, re
import torch
import numpy as np
import soundfile as sf
import runpod

# ── Paths (set via env vars in RunPod) ────────────────────────────────────────
MODELS_DIR   = os.environ.get("MODELS_DIR", "/models")
STT_DIR      = os.path.join(MODELS_DIR, "whisper-darija-ct2")
TTS_CKPT     = os.path.join(MODELS_DIR, "tts", "model_last.pt")
VOCAB_PATH   = os.path.join(MODELS_DIR, "tts", "vocab.txt")
REFS_DIR     = os.path.join(MODELS_DIR, "tts", "refs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {DEVICE}")

# ── Voice reference map ───────────────────────────────────────────────────────
VOICE_REFS = {
    "aisha_happy":    ("darija_001083.wav", "انا هنا خلاصش بعد قلت لدارنا نجي معاكم ما بقاتش معيشة هنا"),
    "aisha_question": ("darija_004964.wav", "ومينيجي موني خاص معاملة خاصة يحظى بها؟"),
    "neutral":        ("darija_000794.wav", "وجبت سيزيام وانتقلت للسيام وكيفاش كيفاش كنت نجيب معدلات مشرفة"),
    "happy":          ("darija_001602.wav", "المراة اللي كانت هي كلش في الدار ولا تسحق مساعدة حتى باش تتحرك"),
    "sad":            ("darija_005314.wav", "قلت احنا كان عندنا الحوش كانت عندنا الارض كانوا عندنا زوايل الدراهم الذهب"),
    "question":       ("darija_004964.wav", "ومينيجي موني خاص معاملة خاصة يحظى بها؟"),
}
REF_TEXTS = {k: v[1] for k, v in VOICE_REFS.items()}

EMOTIONS = {"professional": 0, "enthusiastic": 1, "assertive": 2, "apologetic": 3, "questioning": 4}

# ── Load STT ──────────────────────────────────────────────────────────────────
print("[STT] Loading faster-whisper...")
from faster_whisper import WhisperModel
_stt_model = WhisperModel(STT_DIR, device="cpu", compute_type="int8")
print("[STT] Ready!")

# ── Load TTS ──────────────────────────────────────────────────────────────────
print("[TTS] Loading F5-TTS...")
sys.path.insert(0, "/app/F5-TTS/src")

import soundfile as _sf
import torchaudio as _torchaudio
def _load_sf(filepath, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = _sf.read(str(filepath), dtype="float32", always_2d=True)
    t = torch.from_numpy(data.T if channels_first else data)
    return t, sr
_torchaudio.load = _load_sf

from f5_tts.model import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import load_vocoder

vocab_char_map, vocab_size = get_tokenizer(VOCAB_PATH, "custom")
transformer = DiT(
    dim=1024, depth=22, heads=16, ff_mult=2,
    text_dim=512, conv_layers=4,
    text_num_embeds=vocab_size,
    text_mask_padding=True,
    pe_attn_head=None,
    num_emotions=5,
)
_tts_model = CFM(
    transformer=transformer,
    mel_spec_kwargs=dict(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos",
    ),
    vocab_char_map=vocab_char_map,
    emotion_drop_prob=0.1,
)
ckpt = torch.load(TTS_CKPT, map_location="cpu", weights_only=True)
state = ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict") or ckpt
cleaned = {k.replace("ema_model.", ""): v for k, v in state.items()
           if k not in ["initted", "update", "step"]}
current = _tts_model.state_dict()
safe = {k: v for k, v in cleaned.items()
        if k not in current or current[k].shape == v.shape}
_tts_model.load_state_dict(safe, strict=False)
_tts_model = _tts_model.to(DEVICE).eval()

_vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=DEVICE)
print("[TTS] Ready!")

# ── Pre-cache voice refs ───────────────────────────────────────────────────────
_voice_cache = {}
def _load_voice(name):
    if name in _voice_cache:
        return _voice_cache[name]
    fname, _ = VOICE_REFS.get(name, VOICE_REFS["aisha_happy"])
    path = os.path.join(REFS_DIR, fname)
    import torchaudio
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    audio_np = audio.squeeze(0).numpy()
    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    if rms < 0.1:
        audio_np = audio_np * (0.1 / max(rms, 1e-8))
    audio = torch.from_numpy(audio_np).unsqueeze(0)
    if audio.shape[-1] > 8 * 24000:
        audio = audio[:, :8 * 24000]
    cond = audio.to(DEVICE)
    _voice_cache[name] = cond
    return cond

print("[VOICE] Pre-caching voice refs...")
for v in VOICE_REFS:
    _load_voice(v)
print("[VOICE] Done!")

# ── STT ───────────────────────────────────────────────────────────────────────
def _transcribe(audio_bytes: bytes) -> str:
    import io as _io
    buf = _io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as F
        t = torch.from_numpy(data).unsqueeze(0)
        data = F.resample(t, sr, 16000).squeeze(0).numpy()
    segments, _ = _stt_model.transcribe(data, language="ar", beam_size=5)
    return "".join(s.text for s in segments).strip()

# ── TTS ───────────────────────────────────────────────────────────────────────
def _synthesize(text: str, emotion: str = "professional", voice: str = "aisha_happy") -> str:
    eid = EMOTIONS.get(emotion, 0)
    cond = _load_voice(voice)

    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in '.؟!،':
        text += '.'

    ref_text  = REF_TEXTS.get(voice, REF_TEXTS["aisha_happy"])
    full_text = ref_text + " " + text

    with torch.no_grad():
        ref_mel_len = _tts_model.mel_spec(cond).shape[-1]

    ref_chars      = max(len(ref_text), 1)
    gen_chars      = max(len(text), 1)
    duration_ratio = int(ref_mel_len / ref_chars * gen_chars)
    total_frames   = ref_mel_len + max(duration_ratio, 50)
    total_frames   = min(total_frames, 8192)

    emo_t = torch.LongTensor([eid]).to(DEVICE)
    with torch.inference_mode():
        generated, _ = _tts_model.sample(
            cond=cond, text=[full_text], duration=total_frames,
            steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
            emotion_id=emo_t,
        )

    generated = generated.to(torch.float32)[:, ref_mel_len:, :]
    if generated.shape[1] == 0:
        return None
    generated = generated.permute(0, 2, 1)
    wav = _vocoder.decode(generated).squeeze(0).cpu().numpy()
    if wav.ndim > 1:
        wav = wav[0]

    fade = int(0.02 * 24000)
    if len(wav) > fade:
        wav[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    wav = wav - wav.mean()
    wav = wav.clip(-1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

# ── RunPod handler ────────────────────────────────────────────────────────────
def handler(job):
    inp    = job["input"]
    action = inp.get("action")

    try:
        if action == "stt":
            audio_b64 = inp["audio"]
            audio_bytes = base64.b64decode(audio_b64)
            text = _transcribe(audio_bytes)
            return {"text": text}

        elif action == "tts":
            text    = inp["text"]
            emotion = inp.get("emotion", "professional")
            voice   = inp.get("voice", "aisha_happy")
            audio   = _synthesize(text, emotion, voice)
            if audio is None:
                return {"error": "TTS generated 0 frames"}
            return {"audio": audio}

        else:
            return {"error": f"Unknown action: {action}"}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
