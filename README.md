# AI-Based Accident Detection System Using Artificial Intelligence and Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Academic Level](https://img.shields.io/badge/Level-2nd%20Year%20Engineering%20Mini--Project-purple.svg)]()

A complete, beginner-friendly computer vision project designed for second-year engineering students (Computer Science / AIML / Information Technology). The application detects traffic vehicles (cars, motorcycles, buses, trucks) frame-by-frame using **YOLOv8** and analyzes physical collision anomalies using spatial Intersection over Union (IoU), multi-frame persistence, and kinematic deceleration heuristics.

---

## 🌟 Key Features

1. **Modern Dark-Blue & Red Accent Dashboard:** Built using Streamlit with custom CSS cards, status ribbons, and incident notification alerts.
2. **Vehicle Detection with YOLOv8:** Localizes and labels cars, motorcycles, buses, and trucks with high precision.
3. **Decoupled Collision Analysis Engine:** Separates vehicle detection from accident inference. Avoids false alarms by verifying spatial overlap across consecutive frames ($T \ge 4$) alongside abrupt deceleration.
4. **Preloaded Demo Scenarios & Custom Uploads:** Test immediately using pre-generated collision or normal traffic videos, or upload real CCTV traffic videos (`.mp4`, `.avi`, `.mov`).
5. **Real-Time Video HUD & Visual Alerts:** Displays bounding boxes, tracking centroids, impact markers, and heads-up telemetry.
6. **Downloadable Outputs:**
   - 🎥 Annotated video file (`.mp4`) with bounding boxes and alert overlays.
   - 📊 CSV Incident Log (`accident_detection_log.csv`) detailing timestamps, frame numbers, vehicle classes, and IoU overlap ratios.
7. **Keyframe Incident Gallery:** Displays captured snapshots of exact collision moments for post-incident review.
8. **Manual Location Identifier:** Allows custom location labeling for audit logging without fabricating GPS coordinates.
9. **Academic Disclaimer:** Prominently warns that predictions are experimental and require human verification.

---

## 📁 Project Directory Structure

```
Accident_Detection_System/
│
├── app.py                      # Main Streamlit web application & interactive UI
├── detector.py                 # YOLOv8 vehicle detector, centroid tracker & collision engine
├── config.py                   # Configuration parameters, thresholds & class mappings
├── utils.py                    # OpenCV video annotation, HUD rendering & synthetic video generator
├── generate_sample_video.py    # Standalone script to create realistic sample test videos
├── requirements.txt            # Python dependencies (pinned versions)
├── run_app.bat                 # One-click Windows launch script
├── README.md                   # Comprehensive project documentation & usage guide
├── project_report.md           # 16-section college-ready project report
├── viva_questions.md           # 18 curated viva voce questions with concise answers
│
├── models/                     # Downloaded model weights (yolov8n.pt auto-cached here)
├── sample_videos/              # Preloaded sample videos (collision & normal traffic scenarios)
└── outputs/                    # Exported annotated videos and incident snapshots
```

---

## ⚙️ Installation & Setup (Windows)

### Prerequisites
* Windows 10 or 11
* Python 3.10 to 3.13 installed ([Download Python](https://www.python.org/downloads/))
* Internet connection (only for initial model download, ~6.2 MB)

### Step 1: Open PowerShell or Command Prompt
Navigate to the project folder:
```powershell
cd "c:\Users\Shree\Desktop\DSA\Accident_Detection_System"
```

### Step 2: Install Required Libraries
```powershell
pip install -r requirements.txt
```

### Step 3: Generate Demonstration Videos
Create the sample videos for immediate testing:
```powershell
python generate_sample_video.py
```

---

## 🚀 Running the Application

### Option A: Using the One-Click Batch Script (Recommended for Windows)
Simply double-click `run_app.bat` or run:
```powershell
.\run_app.bat
```

### Option B: Using Streamlit Command
```powershell
streamlit run app.py
```
The browser will automatically launch at `http://localhost:8501`.

---

## 👨‍🏫 How to Demonstrate the Project to Your Teacher

Follow this structured 3-minute demonstration plan:

1. **Introduction (30 seconds):**
   * Introduce yourself and state your project title: *"AI-Based Accident Detection System Using Artificial Intelligence and Computer Vision."*
   * Explain the motivation: Road accidents require rapid alerting, but human CCTV monitoring is error-prone and fatiguing.
2. **Explain the Technology Stack (45 seconds):**
   * **YOLOv8 (Ultralytics):** Performs real-time multi-class vehicle detection on CPU.
   * **OpenCV:** Decodes video frame-by-frame and renders visual HUD annotations.
   * **Decoupled Collision Engine:** Emphasize that **vehicle detection is decoupled from collision classification**. The collision engine checks for persistent spatial overlap ($\text{IoU} \ge 0.18$ sustained over 4+ frames) and abrupt velocity drops.
3. **Live Demonstration (1 minute):**
   * On Tab 1 (**Video Analysis & Live Detection**), select **Preloaded Scenario: Collision Scenario**.
   * Click **▶️ Start Video Analysis**.
   * Point out the real-time bounding boxes, vehicle tracking IDs, and the red **🚨 POSSIBLE ACCIDENT DETECTED** banner when vehicles collide.
   * Switch to the **Smooth Flow Scenario** and run it to demonstrate that no false alerts are raised during normal traffic flow.
4. **Download and Review Reports (30 seconds):**
   * Click **Download Detection Log (.CSV)** and open the generated spreadsheet to show the automated audit trail.
   * Highlight the **Collision Moment Keyframes** gallery showing snapshots of the accident.
5. **Conclusion & Academic Integrity (15 seconds):**
   * Show the prominent disclaimer stating that AI alerts are experimental decision-support aids requiring certified human operator verification.

---

## 🧪 Testing Scenarios

| Test Case | Scenario | Expected Behavior | Status |
|---|---|---|---|
| TC-01 | Sample Collision Video | Detects vehicles, triggers red accident alert at collision moment, captures keyframes | Pass |
| TC-02 | Sample Normal Traffic | Detects vehicles, maintains green status, zero collision alerts | Pass |
| TC-03 | Custom MP4/AVI Upload | Reads custom video file, validates codec, processes frame-by-frame | Pass |
| TC-04 | CSV Log Generation | Generates structured CSV with timestamps, vehicle IDs, and IoU values | Pass |
| TC-05 | Invalid File Handling | Displays graceful error message when non-video or corrupt file is loaded | Pass |

---

## 🔍 How to Use External Road Traffic Videos
You can test the system on real highway and urban traffic footage:
1. Obtain any traffic CCTV clip in MP4, AVI, or MOV format (e.g., from Kaggle Road Traffic Dataset or YouTube Creative Commons).
2. In the dashboard sidebar, select **Upload Custom Traffic Video**.
3. Adjust the **Vehicle Confidence Threshold** (0.30 - 0.45) depending on video clarity.
4. Adjust the **Collision IoU Overlap Threshold** (0.15 - 0.25) depending on camera angle (higher camera angles give better separation).
5. Click **▶️ Start Video Analysis**.

---

## ⚠️ Academic Disclaimer
*This application is an educational prototype developed for a second-year engineering mini-project. All collision alerts are based on geometric overlap and kinematic heuristics. This software does not contact emergency services and must never replace official emergency response systems.*
