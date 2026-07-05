import numpy as np
import librosa
import torch
import tempfile
import os
import soundfile as sf
from transformers import (
    pipeline,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
from models import ToneResult


# ── Model ─────────────────────────────────────────────────────────────────────
#
# WHY THIS MODEL:
#   "superb/wav2vec2-base-superb-er" is the standard benchmark model for
#   Speech Emotion Recognition. Unlike "ehcalabres/wav2vec2-lg-xlsr-en-*"
#   it has a CORRECT classifier head and loads cleanly with the
#   audio-classification pipeline (no UNEXPECTED/MISSING weight warnings).
#
#   Labels: ang (angry) | hap (happy) | neu (neutral) | sad
#   These 4 are the IEMOCAP canonical set — reliable on real conversational speech.
#
_MODEL_ID = "superb/wav2vec2-base-superb-er"

# Map short labels → readable names used in ToneResult
_LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    # superb sometimes returns full names too — keep both
    "angry":   "angry",
    "happy":   "happy",
    "neutral": "neutral",
    "sad":     "sad",
}

_CANONICAL_LABELS = ["neutral", "happy", "sad", "angry"]


class ToneAnalyzer:
    def __init__(self):
        # Load feature extractor + model separately so we control sampling rate
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(_MODEL_ID)
        self._model = AutoModelForAudioClassification.from_pretrained(_MODEL_ID)
        self._model.eval()
        self._device = 0 if torch.cuda.is_available() else -1

        # pipeline as thin wrapper — we pass numpy arrays directly
        self.ser_pipeline = pipeline(
            task="audio-classification",
            model=self._model,
            feature_extractor=self._feature_extractor,
            device=self._device,
        )

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def analyze(self, audio_bytes: bytes) -> ToneResult:
        try:
            y, sr = self._load_audio(audio_bytes)

            tone_features = self._extract_tone_features(y, sr)
            emotion_result = self._classify_emotion_chunked(y, sr)

            return ToneResult(
                success=True,
                dominant_emotion=emotion_result["dominant_emotion"],
                emotion_scores=emotion_result["emotion_scores"],
                pitch_mean=tone_features["pitch_mean"],
                pitch_std=tone_features["pitch_std"],
                energy_mean=tone_features["energy_mean"],
                speaking_rate=tone_features["speaking_rate"],
            )

        except Exception as e:
            return ToneResult(
                success=False,
                message=str(e),
                dominant_emotion="unknown",
                emotion_scores={},
                pitch_mean=0.0,
                pitch_std=0.0,
                energy_mean=0.0,
                speaking_rate=0.0,
                strain_score=0.0,
            )

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _load_audio(self, audio_bytes: bytes):
        """Load raw audio bytes → mono 16kHz numpy array."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        finally:
            os.unlink(tmp_path)
        return y, sr

    def _extract_tone_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract pitch, energy, and speaking-rate features."""
        # Pitch (F0) via pYIN
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_f0  = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std  = float(np.std(voiced_f0))  if len(voiced_f0) > 0 else 0.0

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))

        # Speaking rate (voiced ratio × fps)
        hop_length    = 512
        fps           = sr / hop_length
        voiced_frames = int(np.sum(voiced_flag)) if voiced_flag is not None else 0
        total_frames  = len(f0) if f0 is not None else 1
        speaking_rate = round((voiced_frames / total_frames) * fps, 2)

        return {
            "pitch_mean":   round(pitch_mean, 2),
            "pitch_std":    round(pitch_std,  2),
            "energy_mean":  round(energy_mean, 4),
            "speaking_rate": speaking_rate,
        }

    def _classify_emotion_chunked(self, y: np.ndarray, sr: int) -> dict:
        """
        Split audio into 5-second chunks, run SER on each, then average.

        Why chunking?
        - The superb model was trained on short utterances (~5-10 s).
        - Long audio from a full answer confuses it → random-looking results.
        - Averaging across chunks gives stable, representative emotion scores.
        """
        chunk_size     = sr * 5   # 5 seconds
        min_chunk_size = sr * 2   # ignore < 2 s tails

        chunks = [
            y[start: start + chunk_size]
            for start in range(0, len(y), chunk_size)
            if len(y[start: start + chunk_size]) >= min_chunk_size
        ]
        if not chunks:
            chunks = [y]   # audio shorter than 2 s → use as-is

        # Accumulate scores per canonical label
        accumulated: dict[str, list[float]] = {lbl: [] for lbl in _CANONICAL_LABELS}

        for chunk in chunks:
            # pipeline accepts {"array": np.ndarray, "sampling_rate": int}
            predictions = self.ser_pipeline(
                {"array": chunk.astype(np.float32), "sampling_rate": sr},
                top_k=None,
            )
            for p in predictions:
                raw   = p["label"].lower().strip()
                canon = _LABEL_MAP.get(raw, raw)
                if canon in accumulated:
                    accumulated[canon].append(float(p["score"]))
                # unknown labels → ignore

        # Average + normalise
        emotion_scores = {}
        for lbl, scores in accumulated.items():
            emotion_scores[lbl] = round(float(np.mean(scores)), 4) if scores else 0.0

        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: round(v / total, 4) for k, v in emotion_scores.items()}

        dominant_emotion = max(emotion_scores, key=emotion_scores.get, default="neutral")

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores":   emotion_scores,
        }

    