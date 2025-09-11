from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
import tempfile
import argparse
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import find_peaks

# ------------------- Helper Functions -------------------
def lowpass(sig, alpha=0.2):
    if len(sig) == 0: return sig
    y = [sig[0]]
    for v in sig[1:]:
        y.append(alpha*v + (1-alpha)*y[-1])
    return y

def hip_center_y(landmarks_px):
    lh = landmarks_px.get(23)
    rh = landmarks_px.get(24)
    if lh and rh: return 0.5*(lh[1]+rh[1])
    return None

def robust_height_px(landmarks_px):
    nose = landmarks_px.get(0)
    heel_l = landmarks_px.get(29)
    heel_r = landmarks_px.get(30)
    if not (nose and (heel_l or heel_r)): return None
    heel_y = max([p[1] for p in [heel_l, heel_r] if p is not None])
    return max(0.0, heel_y - nose[1])

def interp(series):
    s = np.array([np.nan if v is None else float(v) for v in series], dtype=float)
    n = len(s)
    idx = np.arange(n)
    if np.all(np.isnan(s)): return s
    good = ~np.isnan(s)
    s[~good] = np.interp(idx[~good], idx[good], s[good])
    return s

def px_to_m_scale(heights_px, real_height_m):
    heights_px = [h for h in heights_px if h is not None]
    if not heights_px: return None
    hpx = np.percentile(heights_px, 90)
    if hpx <= 0: return None
    return real_height_m / hpx

# ------------------- Jump Detection Logic -------------------
def detect_jumps(video_path, real_height_m):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        enable_segmentation=False, min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hip_y, person_h_px = [], []

    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        lm_px = {}
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                x = int(lm.x * W)
                y = int(lm.y * H)
                if 0 <= x < W and 0 <= y < H and lm.visibility > 0.5:
                    lm_px[i] = (x, y)

        if lm_px:
            hip_y.append(hip_center_y(lm_px))
            person_h_px.append(robust_height_px(lm_px))
        else:
            hip_y.append(None)
            person_h_px.append(None)

    cap.release()
    pose.close()

    hip_y_s = np.array(lowpass(interp(hip_y), 0.15))
    person_h_px = [p if p is not None else np.nan for p in person_h_px]

    scale = px_to_m_scale(person_h_px, real_height_m)
    if scale is None:
        return {"error": "Failed to calibrate scale. Ensure first seconds include full upright stance."}

    ground_hip = np.median(hip_y_s[:int(fps)])

    inverted_hip = ground_hip - hip_y_s
    min_distance = int(0.2*fps)
    peaks, _ = find_peaks(inverted_hip, distance=min_distance, prominence=5)

    MIN_HIP_RISE_M = 0.1

    results = []
    for apex_idx in peaks:
        s = apex_idx
        while s > 0 and hip_y_s[s] < ground_hip - 0.05*(hip_y_s.max()-hip_y_s.min()):
            s -= 1
        e = apex_idx
        while e < len(hip_y_s)-1 and hip_y_s[e] < ground_hip - 0.05*(hip_y_s.max()-hip_y_s.min()):
            e += 1
        dh_hip_m = max(0.0, (ground_hip - hip_y_s[apex_idx]) * scale)
        if dh_hip_m < MIN_HIP_RISE_M:
            continue
        results.append({
            "start_time_s": round(s/fps, 2),
            "end_time_s": round(e/fps, 2),
            "hip_rise_m": round(dh_hip_m, 3)
        })

    best = max((r["hip_rise_m"] for r in results), default=0.0)

    return {
        "results": results,
        "best_jump_m": best,
        "scale_m_per_px": round(scale, 6),
        "fps": round(fps, 2)
    }

# ------------------- Flask App -------------------
app = Flask(__name__)
CORS(app)  

@app.route('/jump', methods=['POST'])
def jump():
    if 'video' not in request.files or 'height_m' not in request.form:
        return jsonify({"error": "Missing video file or height parameter"}), 400

    video_file = request.files['video']
    try:
        height_m = float(request.form['height_m'])
    except ValueError:
        return jsonify({"error": "Invalid height_m value"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    try:
        result = detect_jumps(video_path, height_m)
    finally:
        os.remove(video_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001, debug=True)
