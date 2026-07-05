from pydantic import BaseModel
from typing import Dict


class ToneResult(BaseModel):
    success: bool
    message: str = ""

    # Emotion classification
    dominant_emotion: str
    emotion_scores: Dict[str, float]   # {"neutral": 0.82, "happy": 0.10, ...}

    # Tone features
    pitch_mean: float       # Hz — average fundamental frequency
    pitch_std: float        # Hz — pitch variation
    energy_mean: float      # RMS — loudness proxy
    speaking_rate: float    # voiced frames/sec

