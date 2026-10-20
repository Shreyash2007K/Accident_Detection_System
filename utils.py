"""
Utility functions for AI-Based Accident Detection System:
1. Frame annotation (bounding boxes, collision highlights, HUD overlay)
2. Synthetic traffic video generation (Normal traffic & Collision scenario)
3. Video encoding and export helpers
4. CSV detection summary generation
"""

import os
import math
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from config import CLASS_COLORS, COLLISION_BOX_COLOR, WARNING_BOX_COLOR


def draw_annotations(
    frame: np.ndarray,
    detections: List[Any],
    alerts: List[Any],
    frame_idx: int,
    fps: float,
    location_name: str = ""
) -> np.ndarray:
    """
    Annotate frame with vehicle boxes, tracking IDs, speed, and accident alerts.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Collect IDs of vehicles in collision
    collision_vehicle_ids = set()
    warning_vehicle_ids = set()

    for alert in alerts:
        if alert.alert_level == "POSSIBLE_COLLISION":
            if alert.vehicle1.track_id is not None:
                collision_vehicle_ids.add(alert.vehicle1.track_id)
            if alert.vehicle2.track_id is not None:
                collision_vehicle_ids.add(alert.vehicle2.track_id)
        elif alert.alert_level == "PROXIMITY_WARNING":
            if alert.vehicle1.track_id is not None:
                warning_vehicle_ids.add(alert.vehicle1.track_id)
            if alert.vehicle2.track_id is not None:
                warning_vehicle_ids.add(alert.vehicle2.track_id)

    # 1. Draw vehicle bounding boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        track_id = det.track_id
        cls_name = det.class_name
        conf = det.confidence

        # Determine color
        if track_id in collision_vehicle_ids:
            box_color = COLLISION_BOX_COLOR
            thickness = 3
            label_prefix = "CRASH ALERT"
        elif track_id in warning_vehicle_ids:
            box_color = WARNING_BOX_COLOR
            thickness = 2
            label_prefix = "NEAR MISS"
        else:
            box_color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            thickness = 2
            label_prefix = cls_name.upper()

        # Draw Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Label tag
        id_str = f"#{track_id}" if track_id is not None else ""
        label_text = f"{label_prefix} {id_str} ({int(conf * 100)}%)"

        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 6), box_color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, tag_y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Draw vehicle centroid dot
        cx, cy = int(det.centroid[0]), int(det.centroid[1])
        cv2.circle(annotated, (cx, cy), 4, box_color, -1)

    # 2. Draw connector lines and highlight collision zones
    for alert in alerts:
        c1 = (int(alert.vehicle1.centroid[0]), int(alert.vehicle1.centroid[1]))
        c2 = (int(alert.vehicle2.centroid[0]), int(alert.vehicle2.centroid[1]))

        line_color = (0, 0, 255) if alert.alert_level == "POSSIBLE_COLLISION" else (0, 165, 255)
        cv2.line(annotated, c1, c2, line_color, 2, cv2.LINE_AA)

        # Draw midpoint impact marker
        mid_x = (c1[0] + c2[0]) // 2
        mid_y = (c1[1] + c2[1]) // 2
        if alert.alert_level == "POSSIBLE_COLLISION":
            cv2.circle(annotated, (mid_x, mid_y), 12, (0, 0, 255), 2)
            cv2.drawMarker(annotated, (mid_x, mid_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

    # 3. Heads-Up Display (HUD) Banner
    # Top status bar background
    cv2.rectangle(annotated, (0, 0), (w, 54), (15, 23, 42), -1)  # Dark slate blue background

    timestamp_str = f"Time: {frame_idx / max(fps, 1.0):.2f}s | Frame: {frame_idx}"
    cv2.putText(annotated, timestamp_str, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 215, 230), 1, cv2.LINE_AA)

    stats_str = f"Vehicles: {len(detections)}"
    if location_name.strip():
        stats_str += f" | Loc: {location_name.strip()}"
    cv2.putText(annotated, stats_str, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 195, 210), 1, cv2.LINE_AA)

    # System Status Indicator on Top Right
    has_collision = any(a.alert_level == "POSSIBLE_COLLISION" for a in alerts)
    has_warning = any(a.alert_level == "PROXIMITY_WARNING" for a in alerts)

    if has_collision:
        badge_bg = (0, 0, 220)  # Red
        badge_text = "POSSIBLE COLLISION DETECTED"
    elif has_warning:
        badge_bg = (0, 140, 255)  # Orange
        badge_text = "PROXIMITY WARNING"
    else:
        badge_bg = (40, 160, 40)  # Green
        badge_text = "TRAFFIC FLOW NORMAL"

    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bx2 = w - 16
    bx1 = bx2 - bw - 20
    cv2.rectangle(annotated, (bx1, 10), (bx2, 44), badge_bg, -1)
    cv2.putText(annotated, badge_text, (bx1 + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Large prominent emergency alert banner at bottom if collision detected
    if has_collision:
        cv2.rectangle(annotated, (0, h - 38), (w, h), (0, 0, 200), -1)
        alert_msg = "ALERT: POTENTIAL ACCIDENT DETECTED - MANUAL VERIFICATION REQUIRED"
        (aw, ah), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(annotated, alert_msg, ((w - aw) // 2, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def generate_sample_traffic_video(output_path: str, scenario: str = "collision", duration_sec: int = 5, fps: int = 24) -> str:
    """
    Generate a clean synthetic video showing road traffic.
    Scenarios:
      - 'collision': Two cars move towards each other, crash at center, decelerate, and trigger accident detection.
      - 'normal': Cars move in parallel lanes at safe distances with smooth traffic flow.
    Returns the file path of generated video.
    """
    width, height = 640, 480
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Road layout
    road_top = 100
    road_bottom = 420
    lane_y1 = 200
    lane_y2 = 300

    # Car dimensions
    car_w, car_h = 75, 45

    # Initial positions
    # Car 1 (blue sedan): travels from left to right along lane 2
    car1_x = 40.0
    car1_y = 230.0
    car1_speed = 4.2

    # Car 2 (red hatchback): travels from right to left or merges
    if scenario == "collision":
        car2_x = 540.0
        car2_y = 245.0
        car2_speed = -4.0
    else:
        car2_x = 540.0
        car2_y = 135.0  # Separate lane
        car2_speed = -3.8

    # Car 3 (green truck/SUV): travels in bottom lane
    car3_x = 80.0
    car3_y = 330.0
    car3_speed = 3.2

    has_collided = False

    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Background Grass/Shoulder
        frame[0:road_top] = (45, 90, 45)      # Top verge
        frame[road_bottom:height] = (45, 90, 45) # Bottom verge

        # 2. Road surface (Asphalt gray)
        frame[road_top:road_bottom] = (60, 60, 65)

        # 3. Road borders
        cv2.line(frame, (0, road_top), (width, road_top), (255, 255, 255), 3)
        cv2.line(frame, (0, road_bottom), (width, road_bottom), (255, 255, 255), 3)

        # 4. Dashed lane dividers
        dash_offset = (frame_idx * 4) % 40
        for x in range(-40 + dash_offset, width + 40, 40):
            cv2.line(frame, (x, lane_y1), (x + 20, lane_y1), (220, 220, 220), 2)
            cv2.line(frame, (x, lane_y2), (x + 20, lane_y2), (220, 220, 220), 2)

        # Update vehicle positions
        if scenario == "collision":
            # Check if Car 1 and Car 2 touch
            dist = math.hypot(car1_x - car2_x, car1_y - car2_y)
            if dist < 50:
                has_collided = True
                car1_speed = 0.0  # Stopped after impact
                car2_speed = 0.0
            else:
                if not has_collided:
                    car1_x += car1_speed
                    car2_x += car2_speed
        else:
            car1_x = (car1_x + car1_speed) % (width + 100)
            car2_x = (car2_x + car2_speed)
            if car2_x < -100:
                car2_x = width + 50

        car3_x = (car3_x + car3_speed) % (width + 100)

        # Render Car 1 (Blue)
        c1_x1, c1_y1 = int(car1_x), int(car1_y)
        cv2.rectangle(frame, (c1_x1, c1_y1), (c1_x1 + car_w, c1_y1 + car_h), (210, 110, 40), -1)
        cv2.rectangle(frame, (c1_x1, c1_y1), (c1_x1 + car_w, c1_y1 + car_h), (10, 10, 10), 2)  # Border
        cv2.rectangle(frame, (c1_x1 + 15, c1_y1 + 8), (c1_x1 + 50, c1_y1 + car_h - 8), (140, 60, 20), -1) # Cabin
        # Wheels
        cv2.circle(frame, (c1_x1 + 12, c1_y1), 4, (10, 10, 10), -1)
        cv2.circle(frame, (c1_x1 + car_w - 12, c1_y1), 4, (10, 10, 10), -1)
        cv2.circle(frame, (c1_x1 + 12, c1_y1 + car_h), 4, (10, 10, 10), -1)
        cv2.circle(frame, (c1_x1 + car_w - 12, c1_y1 + car_h), 4, (10, 10, 10), -1)

        # Render Car 2 (Red/Maroon)
        c2_x1, c2_y1 = int(car2_x), int(car2_y)
        cv2.rectangle(frame, (c2_x1, c2_y1), (c2_x1 + car_w, c2_y1 + car_h), (50, 50, 200), -1)
        cv2.rectangle(frame, (c2_x1, c2_y1), (c2_x1 + car_w, c2_y1 + car_h), (10, 10, 10), 2)  # Border
        cv2.rectangle(frame, (c2_x1 + 18, c2_y1 + 8), (c2_x1 + 52, c2_y1 + car_h - 8), (30, 30, 130), -1)
        # Wheels
        cv2.circle(frame, (c2_x1 + 12, c2_y1), 4, (10, 10, 10), -1)
        cv2.circle(frame, (c2_x1 + car_w - 12, c2_y1), 4, (10, 10, 10), -1)
        cv2.circle(frame, (c2_x1 + 12, c2_y1 + car_h), 4, (10, 10, 10), -1)
        cv2.circle(frame, (c2_x1 + car_w - 12, c2_y1 + car_h), 4, (10, 10, 10), -1)

        # If collision happened, draw impact flash
        if has_collided:
            imp_x = int((c1_x1 + car_w + c2_x1) / 2)
            imp_y = int((c1_y1 + c2_y1) / 2 + car_h / 2)
            cv2.circle(frame, (imp_x, imp_y), 18, (0, 220, 255), -1)
            cv2.circle(frame, (imp_x, imp_y), 8, (255, 255, 255), -1)

        # Render Car 3 (Green Truck/Van in lane 3)
        c3_x1, c3_y1 = int(car3_x), int(car3_y)
        cv2.rectangle(frame, (c3_x1, c3_y1), (c3_x1 + car_w + 25, c3_y1 + car_h + 8), (50, 160, 80), -1)
        cv2.rectangle(frame, (c3_x1 + car_w - 5, c3_y1 + 6), (c3_x1 + car_w + 20, c3_y1 + car_h + 2), (30, 90, 40), -1)

        out.write(frame)

    out.release()
    return output_path


def export_detection_summary_csv(incidents: List[Any], total_frames: int, fps: float) -> str:
    """
    Convert recorded incidents into CSV format.
    """
    if not incidents:
        # Create empty template
        df = pd.DataFrame(columns=[
            "Timestamp (s)", "Frame Index", "Vehicle 1", "Vehicle 2",
            "IoU", "Overlap Ratio", "Persistence Frames", "Alert Level", "Heuristic Score", "Details"
        ])
    else:
        records = []
        for inc in incidents:
            records.append({
                "Timestamp (s)": inc.timestamp_sec,
                "Frame Index": inc.frame_idx,
                "Vehicle 1": f"{inc.vehicle1.class_name} #{inc.vehicle1.track_id}",
                "Vehicle 2": f"{inc.vehicle2.class_name} #{inc.vehicle2.track_id}",
                "IoU": inc.iou,
                "Overlap Ratio": inc.overlap_ratio,
                "Persistence Frames": inc.persistence_count,
                "Alert Level": inc.alert_level,
                "Heuristic Score": inc.confidence_heuristic,
                "Details": inc.reason
            })
        df = pd.DataFrame(records)

    return df.to_csv(index=False)
