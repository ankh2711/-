import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

input_video = 'video.mp4'
output_video = 'tracked_output.mp4'
json_file = 'track_data.json'
conf_thresh = 0.3
keypoint_thresh = 0.3
pose_edges = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")

yolo_det = YOLO("yolo12x.pt").to(device)
yolo_pose = YOLO("yolo11x-pose.pt").to(device)
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
track_data = {}

def detect_people(frame):
    results = yolo_det(frame, device=device, imgsz=1280)[0]
    return [
        ([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(conf), "person")
        for x1, y1, x2, y2, conf, cls in results.boxes.data
        if int(cls) == 0 and conf > conf_thresh
    ]

def get_pose(roi):
    if roi.shape[0] < 30 or roi.shape[1] < 30:
        return None
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = yolo_pose(roi_rgb, device=device, imgsz=640, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None
    kp = result.keypoints.data[0].cpu().numpy()
    return [
        [int(x), int(y)] if conf > keypoint_thresh else [-1, -1]
        for x, y, conf in kp
    ]

def draw_skeleton(frame, keypoints, dx=0, dy=0):
    for x, y in keypoints:
        if x != -1:
            cv2.circle(frame, (x + dx, y + dy), 3, (0, 0, 255), -1)
    for a, b in pose_edges:
        if keypoints[a] != [-1, -1] and keypoints[b] != [-1, -1]:
            p1 = (keypoints[a][0] + dx, keypoints[a][1] + dy)
            p2 = (keypoints[b][0] + dx, keypoints[b][1] + dy)
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

def analyze_behavior(kp):
    sitting = (
        kp[11][1] != -1 and kp[13][1] != -1 and kp[15][1] != -1 and
        abs(kp[11][1] - kp[13][1]) < abs(kp[13][1] - kp[15][1]) * 0.75
    )
    gesturing = (
        (kp[7][1] != -1 and kp[5][1] != -1 and kp[7][1] < kp[5][1]) or
        (kp[8][1] != -1 and kp[6][1] != -1 and kp[8][1] < kp[6][1])
    )
    return {"sitting": sitting, "gesturing": gesturing}

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_people(frame)
    tracks = tracker.update_tracks(detections, frame=frame)
    current = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        roi = frame[y1:y2, x1:x2]
        keypoints = get_pose(roi) or [[-1, -1]] * 17
        flags = analyze_behavior(keypoints) if keypoints else {}

        draw_skeleton(frame, keypoints, x1, y1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current.append({
            "id": int(track_id),
            "bbox": [x1, y1, x2, y2],
            "keypoints": keypoints,
            "flags": flags
        })

    track_data[f"frame_{frame_idx:05d}"] = {
        "timestamp": round(frame_idx / fps, 2),
        "people": current
    }

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
with open(json_file, 'w') as f:
    json.dump(track_data, f, indent=2)

print(f"Видео сохранено: {output_video}")
print(f"Данные сохранены: {json_file}")