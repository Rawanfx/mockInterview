import whisper
import tempfile
import os
from models import SpeechResult


class SpeechAnalyzer:
    def __init__(self):
        # "base" is faster and still accurate enough for WPM/pauses.
        # Switch to "small" or "medium" if you need better Arabic support.
        self.model = whisper.load_model("small")

    def transcribe(self, audio_bytes: bytes) -> SpeechResult:
        try:
            with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            result = self.model.transcribe(tmp_path, fp16=False)
            os.unlink(tmp_path)

            metrics = self._extract_metrics(result)

            return SpeechResult(
                success=True,
                text=result["text"].strip(),
                language=result["language"],
                words_per_minute=metrics["words_per_minute"],
                pause_count=metrics["pause_count"],
                clarity_score=metrics["clarity_score"],
                speech_pace=metrics["speech_pace"],
            )

        except Exception as e:
            return SpeechResult(
                success=False,
                message=str(e),
                text="",
                language="unknown",
                words_per_minute=0,
                pause_count=0,
                clarity_score=0,
                speech_pace="unknown",
            )

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _extract_metrics(self, result: dict) -> dict:
        segments = result.get("segments", [])

        if not segments:
            return {
                "words_per_minute": 0,
                "pause_count": 0,
                "clarity_score": 0,
                "speech_pace": "unknown",
            }

        # ── WPM ──────────────────────────────────
        total_duration = segments[-1]["end"] - segments[0]["start"]
        total_words = sum(len(s["text"].split()) for s in segments)

        wpm = (
            round(total_words / total_duration * 60, 2)
            if total_duration > 0
            else 0
        )

        # ── Pace label ───────────────────────────
        if wpm < 70:
            pace = "too_slow"
        elif wpm <= 160:
            pace = "normal"
        elif wpm <= 190:
            pace = "slightly_fast"
        else:
            pace = "too_fast"

        # ── Pauses (gaps > 2 s between segments) ─
        pause_count = sum(
            1
            for i in range(1, len(segments))
            if segments[i]["start"] - segments[i - 1]["end"] > 2
        )

        # ── Clarity ──────────────────────────────
        clarity_score = self._calculate_clarity(segments)

        return {
            "words_per_minute": wpm,
            "pause_count": pause_count,
            "clarity_score": clarity_score,
            "speech_pace": pace,
        }

    def _calculate_clarity(self, segments: list) -> float:
        """
        FIX: avg_logprob is negative. Closer to 0 = clearer speech.
        Typical range: -0.2 (very clear) to -1.0+ (unclear/noise).

        We map it to 0-100 using a clamped linear scale:
          -0.0  → 100
          -0.2  →  80   (good interview speech)
          -0.5  →  50
          -1.0  →   0
          < -1.0 → clamped to 0

        We average the RAW (negative) logprobs, not their abs values.
        """
        raw_logprobs = [s.get("avg_logprob", -1.0) for s in segments]
        avg_logprob = sum(raw_logprobs) / len(raw_logprobs)  # negative number

        # Map [-1.0, 0.0] → [0, 100]; clamp outside this range
        score = (avg_logprob + 1.0) * 100.0   # -1→0, 0→100
        return round(max(0.0, min(100.0, score)), 2)