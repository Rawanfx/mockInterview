import argparse
import json
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RunningMode = vision.RunningMode
VISIBILITY_THRESHOLD = 0.5

# Pose landmark indices (BlazePose, 33 points)
POSE_IDX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}

# Face mesh indices
NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263

# ARKit blendshapes
BLINK_BLENDSHAPES = ["eyeBlinkLeft", "eyeBlinkRight"]

# Brow tension: browInnerUp is the primary anxiety/tension indicator
BROW_TENSION_BLENDSHAPES = [
    "browDownLeft",   # anger
    "browDownRight",
    "browInnerUp",    # worry / nervousness
]

# Fingertip indices only (more accurate than palm centroid)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def rotation_matrix_to_euler_angles(R: np.ndarray):
    """Returns (pitch, yaw, roll) in degrees from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return tuple(math.degrees(a) for a in (x, y, z))


def angle_from_horizontal(p_left, p_right) -> float:
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    return math.degrees(math.atan2(dy, dx))


def angle_from_vertical(p_top, p_bottom) -> float:
    dx = p_bottom[0] - p_top[0]
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))


def blendshape_score(blendshapes, names) -> Optional[float]:
    if not blendshapes:
        return None
    lookup = {c.category_name: c.score for c in blendshapes}
    vals = [lookup[n] for n in names if n in lookup]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Per-frame raw metrics container
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    timestamp: float
    face_detected: bool = False
    pose_detected: bool = False
    hand_detected: bool = False
    blink_score: Optional[float] = None
    is_blink_frame: bool = False
    brow_tension_score: Optional[float] = None
    looking_at_camera: Optional[bool] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    shoulder_tilt_deg: Optional[float] = None
    torso_lean_deg: Optional[float] = None
    head_x: Optional[float] = None
    head_y: Optional[float] = None
    face_scale: Optional[float] = None
    hand_to_face_ratio: Optional[float] = None
    is_face_touch: bool = False


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class BodyLanguageAnalyzer:
    def __init__(
        self,
        pose_model_path: str,
        face_model_path: str,
        hand_model_path: str,
        calibration_seconds: float = 5.0,
        window_seconds: float = 1.0,
        blink_score_threshold: float = 0.35,      # sensible default for eyeBlinkLeft/Right
        blink_min_consec_frames: int = 2,
        gaze_yaw_threshold_deg: float = 20.0,
        gaze_pitch_threshold_deg: float = 15.0,
        face_touch_distance_ratio: float = 2.5,
        posture_deviation_threshold_deg: float = 10.0,
        process_every_n_frames: int = 1,
    ):
        self.pose_model_path = pose_model_path
        self.face_model_path = face_model_path
        self.hand_model_path = hand_model_path
        self.calibration_seconds = calibration_seconds
        self.window_seconds = window_seconds
        self.blink_score_threshold = blink_score_threshold
        self.blink_min_consec_frames = blink_min_consec_frames
        self.gaze_yaw_threshold_deg = gaze_yaw_threshold_deg
        self.gaze_pitch_threshold_deg = gaze_pitch_threshold_deg
        self.face_touch_distance_ratio = face_touch_distance_ratio
        self.posture_deviation_threshold_deg = posture_deviation_threshold_deg
        self.process_every_n_frames = max(1, process_every_n_frames)

    # ------------------------------------------------------------------
    def _build_landmarkers(self):
        pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.pose_model_path),
                running_mode=RunningMode.VIDEO,
            )
        )
        face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.face_model_path),
                running_mode=RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
            )
        )
        return pose, face, hand

    # ------------------------------------------------------------------
    # Calibrate blink threshold per-person from the first N seconds
    # ------------------------------------------------------------------
    def _calibrate_blink_threshold(
            self, frames: list[FrameMetrics]) -> float:
        """
        FIX: The eyeBlinkLeft/Right blendshape is HIGH when eye is CLOSED
        (approaching 1.0 = fully closed) and LOW when eye is open (≈0.0–0.2).

        Strategy:
          1. Collect blink scores from the first 10 s (mostly open-eye baseline).
          2. Compute mean of open-eye scores.
          3. Set threshold = mean + 1.5 * std  → catches spikes above normal open-eye level.
          4. Clamp to [0.25, 0.70] for safety.

        This means "is_closed = blink_score >= threshold" is correct:
        a spike in the blink score above the open-eye baseline = blink.
        """
        cutoff = 10.0
        scores = [
            f.blink_score
            for f in frames
            if f.timestamp <= cutoff and f.blink_score is not None
        ]
        if len(scores) < 10:
            return self.blink_score_threshold  # not enough data → fallback

        mean = float(np.mean(scores))
        std = float(np.std(scores))

        # Open-eye scores are low (≈0.05–0.15). A blink = spike above that.
        # mean + 1.5*std gives a threshold that is clearly above normal noise.
        threshold = mean + 1.5 * std

        # Clamp: never lower than 0.25 (avoid noise triggers),
        #        never higher than 0.70 (would miss real blinks).
        return float(np.clip(threshold, 0.25, 0.70))

    # ------------------------------------------------------------------
    # Classify head movement as stable / natural / nervous
    # ------------------------------------------------------------------
    def _classify_head_movement(
            self, displacements: list[float]) -> str:
        """
        Distinguish between:
          stable  — barely any movement
          natural — occasional deliberate nods / turns
          nervous — frequent small rapid movements
        """
        if not displacements:
            return "stable"

        mean_disp = float(np.mean(displacements))
        rapid_moves = sum(1 for d in displacements if d > 0.05)
        frequency = rapid_moves / len(displacements)

        if mean_disp < 0.02:
            return "stable"
        elif frequency > 0.3:
            return "nervous"
        else:
            return "natural"

    # ------------------------------------------------------------------
    def process_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        pose_lm, face_lm, hand_lm = self._build_landmarkers()

        raw_frames: list[FrameMetrics] = []
        blink_timestamps: list[float] = []

        # ── Pass 1: collect all frames ───────────────────────────────
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % self.process_every_n_frames != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / fps
                timestamp_ms = int(timestamp * 1000)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb)

                pose_result = pose_lm.detect_for_video(
                    mp_image, timestamp_ms)
                face_result = face_lm.detect_for_video(
                    mp_image, timestamp_ms)
                hand_result = hand_lm.detect_for_video(
                    mp_image, timestamp_ms)

                fm = FrameMetrics(timestamp=timestamp)

                # ── Face ──────────────────────────────────────────────
                if face_result.face_landmarks:
                    fm.face_detected = True
                    fl = face_result.face_landmarks[0]

                    blendshapes = (
                        face_result.face_blendshapes[0]
                        if face_result.face_blendshapes else None
                    )
                    fm.blink_score = blendshape_score(
                        blendshapes, BLINK_BLENDSHAPES)

                    fm.brow_tension_score = blendshape_score(
                        blendshapes, BROW_TENSION_BLENDSHAPES)

                    fm.face_scale = euclidean(
                        (fl[LEFT_EYE_OUTER_IDX].x * w,
                         fl[LEFT_EYE_OUTER_IDX].y * h),
                        (fl[RIGHT_EYE_OUTER_IDX].x * w,
                         fl[RIGHT_EYE_OUTER_IDX].y * h),
                    )
                    fm.head_x = fl[NOSE_TIP_IDX].x * w
                    fm.head_y = fl[NOSE_TIP_IDX].y * h

                    if face_result.facial_transformation_matrixes:
                        matrix = (
                            face_result.facial_transformation_matrixes[0])
                        rotation = matrix[:3, :3]
                        pitch, yaw, _roll = (
                            rotation_matrix_to_euler_angles(rotation))
                        fm.yaw, fm.pitch = yaw, pitch
                        fm.looking_at_camera = (
                            abs(yaw) <= self.gaze_yaw_threshold_deg
                            and abs(pitch) <= self.gaze_pitch_threshold_deg
                        )

                # ── Pose ──────────────────────────────────────────────
                if pose_result.pose_landmarks:
                    pl = pose_result.pose_landmarks[0]

                    def vis_ok(i):
                        v = pl[i].visibility
                        return v is None or v >= VISIBILITY_THRESHOLD

                    if (vis_ok(POSE_IDX["left_shoulder"])
                            and vis_ok(POSE_IDX["right_shoulder"])):
                        fm.pose_detected = True
                        ls = (pl[POSE_IDX["left_shoulder"]].x * w,
                              pl[POSE_IDX["left_shoulder"]].y * h)
                        rs = (pl[POSE_IDX["right_shoulder"]].x * w,
                              pl[POSE_IDX["right_shoulder"]].y * h)
                        fm.shoulder_tilt_deg = angle_from_horizontal(
                            ls, rs)

                        if (vis_ok(POSE_IDX["left_hip"])
                                and vis_ok(POSE_IDX["right_hip"])):
                            lh = (pl[POSE_IDX["left_hip"]].x * w,
                                  pl[POSE_IDX["left_hip"]].y * h)
                            rh = (pl[POSE_IDX["right_hip"]].x * w,
                                  pl[POSE_IDX["right_hip"]].y * h)
                            shoulder_mid = (
                                (ls[0] + rs[0]) / 2,
                                (ls[1] + rs[1]) / 2,
                            )
                            hip_mid = (
                                (lh[0] + rh[0]) / 2,
                                (lh[1] + rh[1]) / 2,
                            )
                            fm.torso_lean_deg = angle_from_vertical(
                                shoulder_mid, hip_mid)

                # ── Hands ─────────────────────────────────────────────
                if hand_result.hand_landmarks:
                    fm.hand_detected = True
                    if (fm.face_detected
                            and fm.head_x is not None
                            and fm.face_scale):
                        min_ratio = None
                        for hand_pts in hand_result.hand_landmarks:
                            fingertips = [
                                hand_pts[i] for i in FINGERTIP_INDICES]
                            cx = float(
                                np.mean([p.x for p in fingertips])) * w
                            cy = float(
                                np.mean([p.y for p in fingertips])) * h
                            dist = euclidean(
                                (cx, cy), (fm.head_x, fm.head_y))
                            ratio = dist / fm.face_scale
                            if min_ratio is None or ratio < min_ratio:
                                min_ratio = ratio
                        if min_ratio is not None:
                            fm.hand_to_face_ratio = min_ratio
                            fm.is_face_touch = (
                                min_ratio <= self.face_touch_distance_ratio)

                raw_frames.append(fm)
                frame_idx += 1

        finally:
            cap.release()
            pose_lm.close()
            face_lm.close()
            hand_lm.close()

        # ── FIX: calibrate blink threshold then re-detect blinks ─────
        calibrated_threshold = self._calibrate_blink_threshold(raw_frames)
        below_threshold_run = 0

        for fm in raw_frames:
            fm.is_blink_frame = False  # reset
            if fm.blink_score is not None:
                # eyeBlinkLeft/Right is HIGH when closed → spike = blink
                is_closed = fm.blink_score >= calibrated_threshold
                if is_closed:
                    below_threshold_run += 1
                else:
                    # Transition: was closed for ≥ N frames → count as blink
                    if below_threshold_run >= self.blink_min_consec_frames:
                        blink_timestamps.append(fm.timestamp)
                        fm.is_blink_frame = True
                    below_threshold_run = 0

        # Baseline uses median (robust to nervous first seconds)
        baseline = self._compute_baseline(raw_frames)
        time_series = self._aggregate_windows(
            raw_frames, blink_timestamps, baseline)
        summary = self._compute_summary(
            time_series, blink_timestamps, raw_frames,
            calibrated_threshold)

        return {
            "fps": fps,
            "duration_seconds": frame_idx / fps if fps else None,
            "calibration_baseline": baseline,
            "calibrated_blink_threshold": calibrated_threshold,
            "time_series": time_series,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Baseline uses median (robust to outliers in first few seconds)
    # ------------------------------------------------------------------
    def _compute_baseline(
            self, frames: list[FrameMetrics]) -> dict:
        cutoff = self.calibration_seconds
        shoulder_vals = [
            f.shoulder_tilt_deg for f in frames
            if f.timestamp <= cutoff
            and f.shoulder_tilt_deg is not None
        ]
        torso_vals = [
            f.torso_lean_deg for f in frames
            if f.timestamp <= cutoff
            and f.torso_lean_deg is not None
        ]

        return {
            "shoulder_tilt_deg": (
                float(np.median(shoulder_vals))
                if shoulder_vals else None),
            "torso_lean_deg": (
                float(np.median(torso_vals))
                if torso_vals else None),
            "samples_used": len(shoulder_vals),
        }

    # ------------------------------------------------------------------
    def _aggregate_windows(
            self, frames, blink_timestamps, baseline) -> list[dict]:
        if not frames:
            return []

        total_duration = frames[-1].timestamp
        n_windows = int(total_duration // self.window_seconds) + 1
        time_series = []
        prev_head_pos = None

        for w_idx in range(n_windows):
            w_start = w_idx * self.window_seconds
            w_end = w_start + self.window_seconds

            window_frames = [
                f for f in frames
                if w_start <= f.timestamp < w_end
            ]
            if not window_frames:
                continue

            looking_flags = [
                f.looking_at_camera for f in window_frames
                if f.looking_at_camera is not None
            ]
            eye_contact_pct = (
                float(np.mean(looking_flags) * 100)
                if looking_flags else None)

            shoulder_vals = [
                f.shoulder_tilt_deg for f in window_frames
                if f.shoulder_tilt_deg is not None
            ]
            torso_vals = [
                f.torso_lean_deg for f in window_frames
                if f.torso_lean_deg is not None
            ]

            shoulder_dev = (
                float(np.mean(shoulder_vals))
                - baseline["shoulder_tilt_deg"]
                if shoulder_vals
                and baseline.get("shoulder_tilt_deg") is not None
                else None
            )
            torso_dev = (
                float(np.mean(torso_vals)) - baseline["torso_lean_deg"]
                if torso_vals
                and baseline.get("torso_lean_deg") is not None
                else None
            )

            poor_posture = (
                (shoulder_dev is not None
                 and abs(shoulder_dev)
                 > self.posture_deviation_threshold_deg)
                or (torso_dev is not None
                    and abs(torso_dev)
                    > self.posture_deviation_threshold_deg)
            )

            displacements = []
            for f in window_frames:
                if f.head_x is not None and f.face_scale:
                    if prev_head_pos is not None:
                        disp = (
                            euclidean(
                                (f.head_x, f.head_y), prev_head_pos)
                            / f.face_scale
                        )
                        displacements.append(disp)
                    prev_head_pos = (f.head_x, f.head_y)

            head_movement_score = (
                float(np.mean(displacements))
                if displacements else None)

            head_movement_type = self._classify_head_movement(displacements)

            brow_vals = [
                f.brow_tension_score for f in window_frames
                if f.brow_tension_score is not None
            ]
            brow_tension = (
                float(np.mean(brow_vals)) if brow_vals else None)

            face_touch_count = sum(
                1 for f in window_frames if f.is_face_touch)
            blinks_in_window = sum(
                1 for t in blink_timestamps
                if w_start <= t < w_end)

            time_series.append({
                "window_start": round(w_start, 2),
                "window_end": round(w_end, 2),
                "eye_contact_pct": eye_contact_pct,
                "shoulder_deviation_deg": shoulder_dev,
                "torso_deviation_deg": torso_dev,
                "poor_posture_flag": poor_posture,
                "head_movement_score": head_movement_score,
                "head_movement_type": head_movement_type,
                "brow_tension_score": brow_tension,
                "face_touch_count": face_touch_count,
                "blink_count": blinks_in_window,
            })

        return time_series

    # ------------------------------------------------------------------
    def _compute_summary(
            self, time_series, blink_timestamps,
            frames, calibrated_threshold: float) -> dict:
        duration_min = (
            frames[-1].timestamp / 60.0) if frames else 0.0

        eye_contact_vals = [
            w["eye_contact_pct"] for w in time_series
            if w["eye_contact_pct"] is not None
        ]
        head_movement_vals = [
            w["head_movement_score"] for w in time_series
            if w["head_movement_score"] is not None
        ]
        brow_vals = [
            w["brow_tension_score"] for w in time_series
            if w["brow_tension_score"] is not None
        ]

        movement_types = [
            w["head_movement_type"] for w in time_series
            if w["head_movement_type"] is not None
        ]
        dominant_movement = (
            max(set(movement_types), key=movement_types.count)
            if movement_types else "stable"
        )

        return {
            "avg_eye_contact_pct": (
                float(np.mean(eye_contact_vals))
                if eye_contact_vals else None),
            "poor_posture_window_pct": (
                float(
                    np.mean([w["poor_posture_flag"]
                             for w in time_series]) * 100)
                if time_series else None),
            "avg_head_movement_score": (
                float(np.mean(head_movement_vals))
                if head_movement_vals else None),
            "dominant_head_movement_type": dominant_movement,
            "avg_brow_tension_score": (
                float(np.mean(brow_vals))
                if brow_vals else None),
            "total_face_touch_events": sum(
                w["face_touch_count"] for w in time_series),
            "blink_rate_per_minute": (
                len(blink_timestamps) / duration_min
                if duration_min > 0 else None),
            "calibrated_blink_threshold": calibrated_threshold,
            "frames_with_face_detected_pct": (
                float(np.mean(
                    [f.face_detected for f in frames]) * 100)
                if frames else None),
            "frames_with_pose_detected_pct": (
                float(np.mean(
                    [f.pose_detected for f in frames]) * 100)
                if frames else None),
            "frames_with_hand_detected_pct": (
                float(np.mean(
                    [f.hand_detected for f in frames]) * 100)
                if frames else None),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mock interview body language analyzer (MediaPipe Tasks API)")
    parser.add_argument("video_path", help="Path to the interview video file")
    parser.add_argument("--pose-model", required=True)
    parser.add_argument("--face-model", required=True)
    parser.add_argument("--hand-model", required=True)
    parser.add_argument("-o", "--output", default="body_language_report.json")
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--process-every-n-frames", type=int, default=1)
    args = parser.parse_args()

    analyzer = BodyLanguageAnalyzer(
        pose_model_path=args.pose_model,
        face_model_path=args.face_model,
        hand_model_path=args.hand_model,
        calibration_seconds=args.calibration_seconds,
        window_seconds=args.window_seconds,
        process_every_n_frames=args.process_every_n_frames,
    )
    result = analyzer.process_video(args.video_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Report written to {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()