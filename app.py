"""
AI-Based Accident Detection System Using Artificial Intelligence and Computer Vision
Author: Second-Year Engineering Project
Tech Stack: Python, Streamlit, OpenCV, YOLOv8, NumPy, Pandas
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Setup directories and imports
from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    ACADEMIC_LEVEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_PERSISTENCE_FRAMES,
    DEFAULT_SPEED_DROP_PERCENT,
    TARGET_VEHICLE_CLASSES,
    SUPPORTED_VIDEO_FORMATS,
    SAMPLE_VIDEOS_DIR,
    OUTPUTS_DIR
)
from detector import VehicleDetector, AccidentAnalysisEngine, VehicleDetection, CollisionAlert
from utils import (
    draw_annotations,
    generate_sample_traffic_video,
    export_detection_summary_csv
)

# Set page configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Mini Project",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-blue & white theme with red accents
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 25px;
        border-left: 6px solid #ef4444;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #cbd5e1;
        font-size: 14px;
        margin: 0;
    }
    .badge-sub {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    /* Stat cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card .value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card .label {
        font-size: 12px;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    /* Alert banners */
    .danger-alert-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 8px;
        color: #991b1b;
        margin: 15px 0;
        font-weight: 500;
    }
    .success-alert-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 14px 18px;
        border-radius: 8px;
        color: #166534;
        margin: 15px 0;
        font-weight: 500;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400e;
        font-size: 13px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "incident_alerts" not in st.session_state:
    st.session_state.incident_alerts = []
if "metrics_summary" not in st.session_state:
    st.session_state.metrics_summary = {}
if "incident_snapshots" not in st.session_state:
    st.session_state.incident_snapshots = []
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


@st.cache_resource
def load_detector_model(model_name: str = "yolov8n.pt"):
    """Cache and load vehicle detector model."""
    try:
        detector = VehicleDetector(model_name_or_path=model_name)
        return detector, None
    except Exception as e:
        return None, str(e)


# Ensure sample videos exist
def ensure_sample_videos():
    collision_demo = SAMPLE_VIDEOS_DIR / "traffic_collision.mp4"
    normal_demo = SAMPLE_VIDEOS_DIR / "traffic_normal.mp4"
    if not collision_demo.exists():
        generate_sample_traffic_video(str(collision_demo), scenario="collision", duration_sec=5, fps=24)
    if not normal_demo.exists():
        generate_sample_traffic_video(str(normal_demo), scenario="normal", duration_sec=5, fps=24)


try:
    ensure_sample_videos()
except Exception as e:
    pass


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    nav_option = st.radio(
        "Navigation",
        [
            "🚀 Video Analysis & Live Detection",
            "📊 Analytics & Incident Logs",
            "🏗️ System Architecture & Theory",
            "🎓 Viva & Demo Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🎛️ Detection Parameters")

    conf_slider = st.slider(
        "Vehicle Confidence Threshold",
        min_value=0.15,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Minimum confidence score required for YOLO to detect a vehicle."
    )

    iou_slider = st.slider(
        "Collision IoU Overlap Threshold",
        min_value=0.10,
        max_value=0.60,
        value=DEFAULT_IOU_THRESHOLD,
        step=0.02,
        help="Bounding box overlap ratio indicating physical contact between vehicles."
    )

    persistence_slider = st.slider(
        "Persistence Frames for Alert",
        min_value=2,
        max_value=10,
        value=DEFAULT_PERSISTENCE_FRAMES,
        step=1,
        help="Number of consecutive overlapping frames before confirming collision (filters momentary overtaking)."
    )

    speed_drop_slider = st.slider(
        "Deceleration Threshold (%)",
        min_value=20,
        max_value=80,
        value=int(DEFAULT_SPEED_DROP_PERCENT),
        step=5,
        help="Percentage velocity drop at collision zone required for abrupt impact heuristic."
    )

    st.markdown("---")
    st.markdown("#### 📍 Location Tagging (Optional Demo Field)")
    location_input = st.text_input(
        "Manual Location Identifier",
        value="Junction 4A - North Expressway",
        help="Demonstration label recorded in audit log. Does not invoke GPS or contact authorities."
    )

    st.markdown("---")
    st.markdown("#### 🚗 Active Vehicle Classes")
    selected_classes = {}
    for cid, cname in TARGET_VEHICLE_CLASSES.items():
        if st.checkbox(f"{cname.capitalize()}", value=True):
            selected_classes[cid] = cname

    st.caption("Developed for College AIML Mini-Project | YOLOv8 + OpenCV")


# Main Dashboard Header
st.markdown(f"""
<div class="main-header">
    <span class="badge-sub">ACADEMIC MINI-PROJECT | {ACADEMIC_LEVEL.upper()}</span>
    <h1>🚨 {APP_TITLE}</h1>
    <p>{APP_SUBTITLE} — Real-time frame processing, vehicle identification, multi-object tracking, and kinematic collision anomaly alerts.</p>
</div>
""", unsafe_allow_html=True)

# Mandatory Experimental Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Academic Prototype & Experimental Advisory:</strong>
    This application is an educational prototype developed for second-year engineering coursework.
    Accident alerts are computed using geometric bounding box Intersection-over-Union (IoU) overlap and kinematic motion heuristics.
    Predictions are probabilistic and require manual verification by human traffic management operators.
</div>
""", unsafe_allow_html=True)


# ==========================================================
# TAB 1: VIDEO ANALYSIS & LIVE DETECTION
# ==========================================================
if nav_option == "🚀 Video Analysis & Live Detection":

    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.subheader("1. Video Source Selection")
        source_mode = st.radio(
            "Select Video Source:",
            ["Use Built-in Demonstration Video", "Upload Custom Traffic Video (MP4, AVI, MOV)"],
            horizontal=True
        )

        video_path = None
        if source_mode == "Use Built-in Demonstration Video":
            demo_type = st.selectbox(
                "Select Preloaded Scenario:",
                [
                    "Accident / Collision Scenario (traffic_collision.mp4)",
                    "Smooth Flow Scenario (traffic_normal.mp4)",
                    "Real Highway CCTV Traffic (real_traffic.mp4)"
                ]
            )
            if "Collision" in demo_type:
                video_path = str(SAMPLE_VIDEOS_DIR / "traffic_collision.mp4")
            elif "Smooth" in demo_type:
                video_path = str(SAMPLE_VIDEOS_DIR / "traffic_normal.mp4")
            else:
                video_path = str(SAMPLE_VIDEOS_DIR / "real_traffic.mp4")
            st.info(f"Loaded demo video: `{os.path.basename(video_path)}`")

        else:
            uploaded_file = st.file_uploader(
                "Upload a traffic video file",
                type=SUPPORTED_VIDEO_FORMATS,
                help="Upload high-angle or CCTV traffic video (MP4, AVI, MOV)."
            )
            if uploaded_file is not None:
                # Save temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                st.success(f"Uploaded `{uploaded_file.name}` successfully!")

    with col_info:
        st.subheader("2. Model & Diagnostics")
        detector, error_msg = load_detector_model()
        if error_msg:
            st.error(f"⚠️ Model Loader Error: {error_msg}")
            st.caption("Please check if `ultralytics` and `torch` are properly installed.")
        else:
            st.success("✅ YOLOv8 Nano Model Ready (CPU Mode)")
            st.markdown("""
            * **Object Model:** YOLOv8n (Pretrained on COCO)
            * **Active Targets:** Cars, Motorcycles, Buses, Trucks
            * **Collision Logic:** Multi-criteria IoU Overlap + Centroid Tracker
            * **Execution Device:** CPU (Beginner friendly)
            """)

    st.markdown("---")

    # Video Inspection and Processing Section
    if video_path and os.path.exists(video_path):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Unable to open selected video file. Please check video codec or upload another file.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or np.isnan(fps):
                fps = 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / max(fps, 1.0)
            cap.release()

            # Video metadata summary
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Video Resolution", f"{width} × {height} px")
            mcol2.metric("Total Frames", f"{total_frames}")
            mcol3.metric("Video Frame Rate", f"{fps:.1f} FPS")
            mcol4.metric("Duration", f"{duration_sec:.2f} seconds")

            st.markdown("### 3. Execution Pipeline")
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                start_btn = st.button("▶️ Start Video Analysis", type="primary", use_container_width=True)

            if start_btn:
                st.session_state.stop_requested = False
                st.session_state.incident_alerts = []
                st.session_state.incident_snapshots = []

                engine = AccidentAnalysisEngine(
                    iou_threshold=iou_slider,
                    overlap_threshold=iou_slider * 1.8,
                    min_persistence=persistence_slider,
                    speed_drop_percent=float(speed_drop_slider)
                )

                cap = cv2.VideoCapture(video_path)
                out_path = str(OUTPUTS_DIR / "annotated_result.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                live_video_frame = st.empty()
                live_alert_banner = st.empty()

                frame_counter = 0
                total_vehicles_seen = set()
                peak_vehicles = 0
                start_time = time.time()

                with st.spinner("Processing video frames with YOLOv8..."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_counter += 1
                        timestamp = frame_counter / max(fps, 1.0)

                        # Run detection
                        raw_detections = detector.detect(frame, conf_threshold=conf_slider)

                        # Filter selected classes
                        filtered_detections = [d for d in raw_detections if d.class_id in selected_classes]

                        # Track peak vehicles
                        if len(filtered_detections) > peak_vehicles:
                            peak_vehicles = len(filtered_detections)

                        # Analyze for collisions
                        tracked_vehicles, active_alerts = engine.analyze_frame(
                            frame_idx=frame_counter,
                            timestamp_sec=timestamp,
                            raw_detections=filtered_detections
                        )

                        for v in tracked_vehicles:
                            if v.track_id is not None:
                                total_vehicles_seen.add(v.track_id)

                        # Record alert snapshots
                        for alert in active_alerts:
                            if alert.alert_level == "POSSIBLE_COLLISION":
                                st.session_state.incident_alerts.append(alert)
                                if len(st.session_state.incident_snapshots) < 6:
                                    # Store snapshot
                                    snap_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    st.session_state.incident_snapshots.append({
                                        "frame": frame_counter,
                                        "time": timestamp,
                                        "image": snap_rgb,
                                        "reason": alert.reason
                                    })

                        # Annotate frame
                        annotated_frame = draw_annotations(
                            frame=frame,
                            detections=tracked_vehicles,
                            alerts=active_alerts,
                            frame_idx=frame_counter,
                            fps=fps,
                            location_name=location_input
                        )

                        out_writer.write(annotated_frame)

                        # Update UI every 2 frames for smooth web playback
                        if frame_counter % 2 == 0 or frame_counter == total_frames:
                            rgb_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            live_video_frame.image(rgb_display, channels="RGB", use_container_width=True)

                            frac = min(1.0, frame_counter / max(total_frames, 1))
                            progress_bar.progress(frac)
                            status_text.text(f"Processed frame {frame_counter} of {total_frames} ({int(frac*100)}%)")

                            # Update live alert banner
                            if any(a.alert_level == "POSSIBLE_COLLISION" for a in active_alerts):
                                live_alert_banner.markdown("""
                                <div class="danger-alert-box">
                                    🚨 <strong>ACTIVE ACCIDENT ALERT DETECTED</strong>: Severe collision overlap detected between vehicles!
                                </div>
                                """, unsafe_allow_html=True)
                            elif any(a.alert_level == "PROXIMITY_WARNING" for a in active_alerts):
                                live_alert_banner.markdown("""
                                <div style="background-color:#fffbeb; border-left:5px solid #f59e0b; padding:12px; border-radius:6px; color:#b45309; font-weight:600; margin:10px 0;">
                                    ⚠️ <strong>NEAR-MISS PROXIMITY WARNING</strong>: Vehicles in close proximity.
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                live_alert_banner.empty()

                cap.release()
                out_writer.release()
                total_duration = time.time() - start_time
                avg_fps = frame_counter / max(total_duration, 0.001)

                st.session_state.analysis_complete = True
                st.session_state.processed_video_path = out_path
                st.session_state.metrics_summary = {
                    "frames_analyzed": frame_counter,
                    "total_vehicles_tracked": len(total_vehicles_seen),
                    "peak_vehicles": peak_vehicles,
                    "alerts_count": len([a for a in st.session_state.incident_alerts if a.alert_level == "POSSIBLE_COLLISION"]),
                    "avg_fps": round(avg_fps, 2),
                    "video_duration_sec": round(duration_sec, 2),
                    "location": location_input
                }

                status_text.text(f"Processing finished in {total_duration:.2f}s (Average {avg_fps:.1f} FPS)")

    # Display Results & Alerts
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.subheader("4. Detection Results & Incident Summary")

        m = st.session_state.metrics_summary
        alert_count = m.get("alerts_count", 0)

        # Alert Banner
        if alert_count > 0:
            st.markdown(f"""
            <div class="danger-alert-box">
                🚨 <strong>POSSIBLE ACCIDENT DETECTED!</strong>
                The system identified <strong>{alert_count}</strong> collision events where spatial bounding box overlap and persistence thresholds were crossed.
                Location Tag: <em>{m.get('location', 'Unspecified')}</em>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-alert-box">
                ✅ <strong>NO ACCIDENT DETECTED</strong>: Traffic flow is smooth and normal. No spatial collisions or anomalous deceleration detected across analyzed frames.
            </div>
            """, unsafe_allow_html=True)

        # Metrics Row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.markdown(f"<div class='metric-card'><div class='value'>{m.get('frames_analyzed', 0)}</div><div class='label'>Frames Analyzed</div></div>", unsafe_allow_html=True)
        kpi2.markdown(f"<div class='metric-card'><div class='value'>{m.get('total_vehicles_tracked', 0)}</div><div class='label'>Vehicles Tracked</div></div>", unsafe_allow_html=True)
        kpi3.markdown(f"<div class='metric-card'><div class='value'>{m.get('peak_vehicles', 0)}</div><div class='label'>Peak Vehicles</div></div>", unsafe_allow_html=True)
        kpi4.markdown(f"<div class='metric-card'><div class='value' style='color:{'#ef4444' if alert_count > 0 else '#22c55e'}'>{alert_count}</div><div class='label'>Collision Alerts</div></div>", unsafe_allow_html=True)
        kpi5.markdown(f"<div class='metric-card'><div class='value'>{m.get('avg_fps', 0)}</div><div class='label'>Average FPS</div></div>", unsafe_allow_html=True)

        st.markdown("### 5. Download Outputs")
        dcol1, dcol2 = st.columns(2)

        # Download CSV
        with dcol1:
            csv_data = export_detection_summary_csv(
                st.session_state.incident_alerts,
                m.get("frames_analyzed", 0),
                fps=24.0
            )
            st.download_button(
                label="📥 Download Detection Log (.CSV)",
                data=csv_data,
                file_name="accident_detection_log.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Download Video
        with dcol2:
            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                with open(st.session_state.processed_video_path, "rb") as vf:
                    st.download_button(
                        label="🎥 Download Annotated Video (.MP4)",
                        data=vf.read(),
                        file_name="annotated_traffic_accident.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

        # Snapshot Gallery
        if st.session_state.incident_snapshots:
            st.markdown("### 6. Collision Moment Keyframes")
            snap_cols = st.columns(min(3, len(st.session_state.incident_snapshots)))
            for idx, snap in enumerate(st.session_state.incident_snapshots[:6]):
                with snap_cols[idx % len(snap_cols)]:
                    st.image(snap["image"], caption=f"Frame {snap['frame']} | Time {snap['time']:.2f}s", use_container_width=True)
                    st.caption(f"Reason: {snap['reason']}")


# ==========================================================
# TAB 2: ANALYTICS & INCIDENT LOGS
# ==========================================================
elif nav_option == "📊 Analytics & Incident Logs":
    st.subheader("📊 Incident Analytics & Verification Audit Trail")

    if not st.session_state.incident_alerts:
        st.info("No video has been analyzed yet or no collision incidents were detected. Please run video analysis on Tab 1.")
    else:
        records = [a.to_dict() for a in st.session_state.incident_alerts]
        df = pd.DataFrame(records)

        st.markdown("#### Complete Incident Log")
        st.dataframe(df, use_container_width=True)

        st.markdown("#### Incident Distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Alert Level Distribution**")
            alert_counts = df["alert_level"].value_counts()
            st.bar_chart(alert_counts)

        with c2:
            st.markdown("**Collision IoU Overlap Over Time**")
            chart_df = df[["timestamp_sec", "iou"]].set_index("timestamp_sec")
            st.line_chart(chart_df)


# ==========================================================
# TAB 3: SYSTEM ARCHITECTURE & THEORY
# ==========================================================
elif nav_option == "🏗️ System Architecture & Theory":
    st.subheader("🏗️ System Architecture & Engineering Methodology")

    st.markdown("""
    ### 1. High-Level System Architecture
    The AI-Based Accident Detection System is designed as a modular pipeline combining state-of-the-art Deep Learning object detection with geometric spatial analysis and kinematics tracking.

    ```
    [Traffic Video Input (MP4/AVI/MOV)]
                    │
                    ▼
       [OpenCV Video Frame Extraction]
                    │
                    ▼
      [YOLOv8 Object Detection Engine]
       (Detects: Cars, Motorcycles, Buses, Trucks)
                    │
                    ▼
     [Centroid Multi-Object Tracker]
       (Calculates: Trajectories & Velocities)
                    │
                    ▼
    [Collision Heuristic & Overlap Analyzer]
       ├─ Step 1: Intersection-over-Union (IoU) calculation
       ├─ Step 2: Overlap Ratio relative to smaller vehicle
       ├─ Step 3: Multi-frame Persistence Filter (T >= 4 frames)
       └─ Step 4: Kinematic Deceleration Verification
                    │
                    ▼
     [Decision & Alert Engine]
       ├─ Normal Flow   -> Green Status HUD
       ├─ Proximity     -> Orange Warning
       └─ Collision     -> 🚨 Red Alert + CSV Incident Log + Video Highlight
    ```

    ### 2. Deep Learning Vehicle Detection (YOLOv8)
    - **Architecture:** You Only Look Once Version 8 (YOLOv8n) developed by Ultralytics.
    - **Backbone & Neck:** Modified CSPDarknet53 with Path Aggregation Network (PAN).
    - **Single-Stage Detection:** Detects bounding coordinates and class probabilities simultaneously in one forward pass.
    - **Target Vehicle Classes:**
      - Car (COCO Class 2)
      - Motorcycle (COCO Class 3)
      - Bus (COCO Class 5)
      - Truck (COCO Class 7)

    ### 3. Collision Analysis Methodology
    - **Intersection-over-Union (IoU):**
      $$IoU = \\frac{\\text{Area}(B_1 \\cap B_2)}{\\text{Area}(B_1 \\cup B_2)}$$
    - **Persistence Filter:** Single-frame overlaps (which occur during normal 3D perspective projection overtaking) are rejected. A collision requires sustained overlap across $\\ge 4$ consecutive frames.
    - **Kinematic Stagnation:** Post-collision vehicles experience an abrupt drop in centroid velocity ($> 40\\%$) and remain trapped within the collision footprint.

    ### 4. Technical Specifications
    | Parameter | Specification |
    |---|---|
    | Programming Language | Python 3.10+ / 3.13 |
    | Object Detection Model | Ultralytics YOLOv8n (Pretrained on COCO) |
    | Computer Vision Library | OpenCV (cv2) |
    | Frontend Framework | Streamlit |
    | Execution Device | CPU (Real-time capable ~15-25 FPS) / GPU Optional |
    | Video Formats | MP4, AVI, MOV, MKV |
    """)


# ==========================================================
# TAB 4: VIVA & DEMO GUIDE
# ==========================================================
elif nav_option == "🎓 Viva & Demo Guide":
    st.subheader("🎓 College Viva Examination & Teacher Demonstration Guide")

    st.markdown("""
    ### 👨‍🏫 How to Demonstrate this Project to your Teacher/Examiner

    1. **Step 1: Introduction (30 seconds)**
       - State your title: *"Respected Professor, our project is titled 'AI-Based Accident Detection System Using Artificial Intelligence and Computer Vision'."*
       - Mention the problem: Manual traffic monitoring cannot scale to millions of CCTV feeds, causing delayed emergency responses to roadway accidents.
    2. **Step 2: Architecture Explanation (1 minute)**
       - Explain that vehicle detection uses **YOLOv8** (You Only Look Once), which performs single-pass feature extraction.
       - Clarify the crucial engineering decision: *Vehicle detection alone does not imply an accident!* We separate vehicle localization from collision reasoning using a multi-criteria kinematic and geometric overlap heuristic.
    3. **Step 3: Live Demonstration (1 minute)**
       - In Tab 1, select **Accident / Collision Scenario** and click **Start Video Analysis**.
       - Show the live bounding boxes, tracking centroids, and the red **POSSIBLE COLLISION** alert banner triggered when vehicles collide.
       - Then switch to **Smooth Flow Scenario** and show how the system maintains a green **TRAFFIC FLOW NORMAL** status without false alarms.
       - Click **Download Detection Log (.CSV)** to showcase automated record keeping.
    4. **Step 4: Academic Integrity Statement (30 seconds)**
       - Emphasize that predictions are clearly marked as experimental and require human operator verification.
    """)

    st.markdown("---")
    st.markdown("### ❓ Top 5 Quick Viva Questions & Answers")

    with st.expander("Q1: Why use YOLO instead of traditional R-CNN for vehicle detection?"):
        st.write("""
        **Answer:** R-CNN is a two-stage detector (generating region proposals first, then classifying each).
        YOLO is a single-stage detector that predicts bounding boxes and class probabilities in a single forward pass,
        making it significantly faster (suitable for real-time video processing at 20-30+ FPS) with lower computational overhead.
        """)

    with st.expander("Q2: How does the system distinguish between normal overtaking and a real accident?"):
        st.write("""
        **Answer:** In normal overtaking, vehicles separate quickly and maintain high velocity.
        Our system uses a **persistence filter** and **kinematic velocity analysis**:
        an accident is only confirmed if the bounding boxes sustain intersection over multiple consecutive frames
        ($\\ge 4$ frames) accompanied by an abrupt velocity drop ($> 40\\%$).
        """)

    with st.expander("Q3: What is Intersection over Union (IoU)?"):
        st.write("""
        **Answer:** IoU is an evaluation and overlap metric calculated as the area of overlap between two bounding boxes
        divided by the area of their union. An IoU of 0 indicates no contact, while values above 0.18-0.25 indicate physical collision or close overlap.
        """)

    with st.expander("Q4: Does this system require an internet connection?"):
        st.write("""
        **Answer:** An internet connection is only needed on the very first run to download the lightweight YOLOv8n weights (~6.2 MB).
        Once downloaded and cached locally, the entire detection and collision pipeline runs completely offline on CPU.
        """)

    with st.expander("Q5: What are the main limitations of this prototype?"):
        st.write("""
        **Answer:**
        1. Camera perspective (2D projection): In single-camera setups, vehicles in different lanes at different depths may appear to overlap visually (occlusion).
        2. Extreme weather: Heavy rain, fog, or night glare can degrade bounding box localization.
        3. Prototype collision heuristic requires human verification before dispatching emergency resources.
        """)
