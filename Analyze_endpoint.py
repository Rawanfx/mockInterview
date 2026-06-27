import os
import subprocess
import tempfile
import json

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict

from models import AnalyzeResponse, BodyLanguageResult


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    video_url: str = Field(alias="videoUrl")
    question_id: int = Field(alias="questionId")


async def _download_video(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Failed to download video: HTTP {resp.status_code}")
        return resp.content


def _extract_audio(video_path: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", audio_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with open(audio_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(audio_path)


async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    video_bytes = await _download_video(request.video_url)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    try:
        from Video_Analysis import BodyLanguageAnalyzer
        body_analyzer = BodyLanguageAnalyzer(
            pose_model_path=os.environ["POSE_MODEL_PATH"],
            face_model_path=os.environ["FACE_MODEL_PATH"],
            hand_model_path=os.environ["HAND_MODEL_PATH"],
        )
        body_result_raw = body_analyzer.process_video(video_path)
        audio_bytes = _extract_audio(video_path)
    finally:
        os.unlink(video_path)

    from speech_analysis import SpeechAnalyzer
    from Tone_analyzer import ToneAnalyzer
    speech_result = SpeechAnalyzer().transcribe(audio_bytes)
    tone_result = ToneAnalyzer().analyze(audio_bytes)

    return AnalyzeResponse(
        success=True,
        question_id=request.question_id,
        body_language=BodyLanguageResult(
            avg_eye_contact_pct=body_result_raw["summary"].get("avg_eye_contact_pct") or 0.0,
            poor_posture_window_pct=body_result_raw["summary"].get("poor_posture_window_pct") or 0.0,
            avg_head_movement_score=body_result_raw["summary"].get("avg_head_movement_score") or 0.0,
            avg_brow_tension_score=body_result_raw["summary"].get("avg_brow_tension_score") or 0.0,
            total_face_touch_events=body_result_raw["summary"].get("total_face_touch_events") or 0,
            blink_rate_per_minute=body_result_raw["summary"].get("blink_rate_per_minute") or 0.0,
            frames_with_face_detected_pct=body_result_raw["summary"].get("frames_with_face_detected_pct") or 0.0,
            frames_with_pose_detected_pct=body_result_raw["summary"].get("frames_with_pose_detected_pct") or 0.0,
            frames_with_hand_detected_pct=body_result_raw["summary"].get("frames_with_hand_detected_pct") or 0.0,
            performance_over_time_json=json.dumps(body_result_raw.get("time_series", [])),
        ),
        speech=speech_result,
        tone=tone_result,
    )