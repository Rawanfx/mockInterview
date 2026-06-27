import asyncio
import tempfile
import os
import subprocess
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from speech_analysis import SpeechAnalyzer
from Tone_analyzer import ToneAnalyzer
from Video_Analysis import BodyLanguageAnalyzer

# ── Model paths (MediaPipe .task files) ──────────────────────────────────────
POSE_MODEL_PATH = "model/pose_landmarker.task"
FACE_MODEL_PATH = "model/face_landmarker.task"
HAND_MODEL_PATH = "model/hand_landmarker.task"

app = FastAPI()

speech_analyzer = None
tone_analyzer = None
body_analyzer = None

@app.on_event("startup")
async def load_models():
    global speech_analyzer, tone_analyzer, body_analyzer
    
    print("Loading models...")
    
    speech_analyzer = SpeechAnalyzer()
    print("✅ Whisper loaded")
    
    tone_analyzer = ToneAnalyzer()
    print("✅ HuggingFace loaded")
    
    body_analyzer = BodyLanguageAnalyzer(
        pose_model_path=POSE_MODEL_PATH,
        face_model_path=FACE_MODEL_PATH,
        hand_model_path=HAND_MODEL_PATH,
    )
    print("✅ MediaPipe loaded")
    
    print("🚀 All models ready!")
# ── Request / Response schemas (تعديلها لتطابق مسميات الـ C#) ─────────────────

class AnalyzeRequest(BaseModel):
    videoUrl: str       # تعديل المسمى لـ camelCase ليطابق C#


# الأجزاء الداخلية من الـ JSON المتوقعة في الـ C# (Nested Objects)
class SpeechTrack(BaseModel):
    text: str           # C# يتوقع result.Speech.Text
    language: str
    speechPace: str
    wordsPerMinute: float
    pauseCount: int
    clarityScore: float

class ToneTrack(BaseModel):
    dominantEmotion: str
    emotionScores: dict
    pitchMean: float
    pitchStd: float
    energyMean: float
    speakingRate: float
    strainScore: float

class BodyLanguageTrack(BaseModel):
    avgEyeContactPct: float | None
    poorPostureWindowPct: float | None
    avgHeadMovementScore: float | None
    avgBrowTensionScore: float | None
    totalFaceTouchEvents: int | None
    blinkRatePerMinute: float | None
    dominantHeadMovementType: str | None = "Unknown"  # مضاف حديثاً في الـ C#
    framesWithFaceDetectedPct: float | None = 0.0
    framesWithPoseDetectedPct: float | None = 0.0
    framesWithHandDetectedPct: float | None = 0.0
    performanceOverTimeJson: str | None = "{}"


class AnalyzeResponse(BaseModel):
    success: bool = True
    message: str = "Analysis completed successfully"
    
    # تحويل المخرجات لـ Objects متطابقة مع شروط كود الـ .NET
    bodyLanguage: BodyLanguageTrack | None
    speech: SpeechTrack | None
    tone: ToneTrack | None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _download_video(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download video: HTTP {response.status_code}",
            )
        return response.content


def _extract_audio(video_path: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ac", "1",
                "-ar", "16000",
                "-vn",
                audio_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(audio_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(audio_path)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    # 1. Download video using camelCase key
    video_bytes = await _download_video(request.videoUrl)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    try:
        # 2. Extract audio
        audio_bytes = _extract_audio(video_path)

        # 3. Run tasks in parallel
        loop = asyncio.get_event_loop()

        speech_task = loop.run_in_executor(None, speech_analyzer.transcribe, audio_bytes)
        tone_task = loop.run_in_executor(None, tone_analyzer.analyze, audio_bytes)
        body_task = loop.run_in_executor(None, body_analyzer.process_video, video_path)

        speech_result, tone_result, body_result = await asyncio.gather(
            speech_task, tone_task, body_task
        )

    finally:
        os.unlink(video_path)

    # 4. Check results
    if not speech_result.success:
        return AnalyzeResponse(success=False, message=f"SpeechAnalyzer failed: {speech_result.message}", bodyLanguage=None, speech=None, tone=None)
    if not tone_result.success:
        return AnalyzeResponse(success=False, message=f"ToneAnalyzer failed: {tone_result.message}", bodyLanguage=None, speech=None, tone=None)

    summary = body_result.get("summary", {})

    # 5. Build response mapping to C# structure exactly
    return AnalyzeResponse(
        success=True,
        message="Success",
        
        speech=SpeechTrack(
            text=speech_result.text,
            language=speech_result.language,
            speechPace=speech_result.speech_pace,
            wordsPerMinute=speech_result.words_per_minute,
            pauseCount=speech_result.pause_count,
            clarityScore=speech_result.clarity_score
        ),
        
        tone=ToneTrack(
            dominantEmotion=tone_result.dominant_emotion,
            emotionScores=tone_result.emotion_scores,
            pitchMean=tone_result.pitch_mean,
            pitchStd=tone_result.pitch_std,
            energyMean=tone_result.energy_mean,
            speakingRate=tone_result.speaking_rate,
            strainScore=tone_result.strain_score
        ),
        
        bodyLanguage=BodyLanguageTrack(
            avgEyeContactPct=summary.get("avg_eye_contact_pct"),
            poorPostureWindowPct=summary.get("poor_posture_window_pct"),
            avgHeadMovementScore=summary.get("avg_head_movement_score"),
            avgBrowTensionScore=summary.get("avg_brow_tension_score"),
            totalFaceTouchEvents=summary.get("total_face_touch_events"),
            blinkRatePerMinute=summary.get("blink_rate_per_minute"),
            dominantHeadMovementType=summary.get("dominant_head_movement_type", "Unknown"),
            framesWithFaceDetectedPct=summary.get("frames_with_face_detected_pct", 0.0),
            framesWithPoseDetectedPct=summary.get("frames_with_pose_detected_pct", 0.0),
            framesWithHandDetectedPct=summary.get("frames_with_hand_detected_pct", 0.0),
            performanceOverTimeJson=summary.get("performance_over_time_json", "{}")
        )
    )

class TranscribeRequest(BaseModel):
    videoUrl: str      


class TranscribeResponse(BaseModel):
    success: bool
    message: str
    text: str   
    
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_quick(request: TranscribeRequest):
    try:
        # 1. تحميل الفيديو من الرابط السحابي
        video_bytes = await _download_video(request.videoUrl)

        # 2. حفظ الفيديو مؤقتاً لاستخراج الصوت منه
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        try:
            # 3. استخراج الصوت بصيغة WAV 16kHz المتوافقة مع Whisper
            audio_bytes = _extract_audio(video_path)

            # 4. استدعاء الـ SpeechAnalyzer لتحويل الصوت إلى نص (تفريغ صوتي)
            # بنشغله في executor عشان الـ Transcription عملية تقيلة ومتقفلش الـ Event Loop
            loop = asyncio.get_event_loop()
            speech_result = await loop.run_in_executor(
                None, speech_analyzer.transcribe, audio_bytes
            )

        finally:
            # مسح ملف الفيديو المؤقت فوراً بعد استخراج الصوت
            os.unlink(video_path)

        # 5. التحقق من نجاح عملية الـ Transcribe
        if not speech_result.success:
            return TranscribeResponse(
                success=False,
                message=f"SpeechAnalyzer failed: {speech_result.message}",
                text=""
            )

        # 6. إرجاع النتيجة بالـ Structure المتوقع في السي شارب
        return TranscribeResponse(
            success=True,
            message="Transcription completed successfully",
            text=speech_result.text  # النص المفرغ
        )

    except Exception as ex:
        return TranscribeResponse(
            success=False,
            message=f"Internal Server Error: {str(ex)}",
            text=""
        )            # C# يتوقع يستقبل result.Text في النهاية