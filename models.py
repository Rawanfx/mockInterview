from pydantic import BaseModel
from typing import Optional, List, Dict
class SpeechResult(BaseModel):
    success: bool
    message: str = ""
    text: str
    language: str
    words_per_minute: float
    pause_count: int
    clarity_score: float
    speech_pace: str
class ToneResult(BaseModel):
    success: bool
    message: str = ""
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    speaking_rate: float
    strain_score: float
class TranscribeResponse(BaseModel):
    success: bool
    text: str
    language: str
    message: str = ""

class BodyLanguageResult(BaseModel):
    avg_eye_contact_pct: float = 0.0
    poor_posture_window_pct: float = 0.0
    avg_head_movement_score: float = 0.0
    avg_brow_tension_score: float = 0.0
    total_face_touch_events: int = 0
    blink_rate_per_minute: float = 0.0
    frames_with_face_detected_pct: float = 0.0
    frames_with_pose_detected_pct: float = 0.0
    frames_with_hand_detected_pct: float = 0.0
    performance_over_time_json: str = ""


class AnalyzeResponse(BaseModel):
    success: bool
    message: str = ""
    question_id: int
    body_language: Optional[BodyLanguageResult] = None
    speech: Optional[SpeechResult] = None
    tone: Optional[ToneResult] = None    

