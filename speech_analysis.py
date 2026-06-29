import tempfile
import os
from groq import Groq
from models import SpeechResult

client = Groq()


class SpeechAnalyzer:
    def transcribe(self, audio_bytes: bytes) -> SpeechResult:
        try:
            with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )

            os.unlink(tmp_path)

            metrics = self._extract_metrics(result)

            return SpeechResult(
                success=True,
                text=result.text.strip(),
                language=result.language,
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

    def _extract_metrics(self, result) -> dict:
        segments = getattr(result, "segments", []) or []

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

        # ── Pauses (gaps > 2s between segments) ──
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
        avg_logprob is negative. Closer to 0 = clearer speech.
        Typical range: -0.2 (very clear) to -1.0+ (unclear/noise).

        Map [-1.0, 0.0] → [0, 100]:
          -0.0  → 100
          -0.2  →  80  (good interview speech)
          -0.5  →  50
          -1.0  →   0
        """
        valid = [
            s for s in segments
            if s.get("avg_logprob") is not None
        ]

        if not valid:
            return self._clarity_fallback(segments)

        raw_logprobs = [s["avg_logprob"] for s in valid]
        avg_logprob = sum(raw_logprobs) / len(raw_logprobs)  # negative number

        score = (avg_logprob + 1.0) * 100.0
        return round(max(0.0, min(100.0, score)), 2)

    def _clarity_fallback(self, segments: list) -> float:
        """
        لو avg_logprob مش موجود لأي سبب —
        بنحسب clarity من عدد الكلمات الغير واضحة.
        """
        if not segments:
            return 0.0

        unclear_markers = ["[inaudible]", "[unclear]", "...", " uh ", " um "]
        unclear_count = sum(
            1 for s in segments
            for marker in unclear_markers
            if marker in s.get("text", "").lower()
        )

        unclear_ratio = unclear_count / len(segments)
        return round(max(0.0, min(100.0, (1 - unclear_ratio) * 100)), 2)