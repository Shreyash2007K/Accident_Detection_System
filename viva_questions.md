# Comprehensive Viva Voce Questions & Answers
## AI-Based Accident Detection System Using Artificial Intelligence and Computer Vision

*Designed for Second-Year Engineering AIML / Computer Science Students*

---

### Section 1: Artificial Intelligence & Machine Learning

#### Q1. What is Artificial Intelligence (AI) and how is it used in this project?
**Answer:** Artificial Intelligence is the broad field of computer science aimed at building systems capable of performing tasks that typically require human intelligence, such as visual perception, pattern recognition, and decision-making. In this project, AI is applied through computer vision models to automatically detect vehicles on roads and determine whether a collision event has occurred without manual human surveillance.

#### Q2. What is Machine Learning (ML) and how does it differ from traditional rule-based programming?
**Answer:** In traditional programming, humans write explicit rules and input data to get an answer. In Machine Learning, an algorithm is fed large amounts of data and corresponding labels; the model learns statistical patterns and weights automatically. Here, machine learning enables the system to recognize various types of vehicles under differing angles and lighting conditions without hardcoding vehicle shapes.

#### Q3. What is Deep Learning and why is it preferred for image tasks?
**Answer:** Deep Learning is a subset of Machine Learning based on multi-layered artificial neural networks (such as Convolutional Neural Networks). Traditional ML requires manual handcrafted feature extraction (e.g., edge detection, color histograms), whereas Deep Learning automatically learns hierarchical representations directly from raw pixel data.

---

### Section 2: Computer Vision & Video Frame Processing

#### Q4. What is Computer Vision (CV)?
**Answer:** Computer Vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos. It converts arrays of raw numbers (pixel intensities) into actionable insights, such as detecting object boundaries, calculating optical motion, and classifying scenes.

#### Q5. How does video processing work in OpenCV?
**Answer:** A digital video is a sequential series of static images called *frames*, rendered at a specific rate measured in *Frames Per Second (FPS)*. OpenCV uses `cv2.VideoCapture` to read and decode a video file frame-by-frame as NumPy 3D arrays (Height × Width × Channels in BGR order), applies detection logic on each array, and displays or writes the annotated frames back using `cv2.VideoWriter`.

#### Q6. What is the difference between RGB and BGR color formats?
**Answer:** Both represent colors using Red, Green, and Blue channels. However, standard Python image libraries (like PIL and Matplotlib) and web browsers expect the channels in **RGB** order, whereas OpenCV historically reads and writes image matrices in **BGR** (Blue, Green, Red) order. Converting between them (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`) is necessary to prevent color inversion during display.

---

### Section 3: Convolutional Neural Networks (CNN) & YOLO

#### Q7. What is a Convolutional Neural Network (CNN)?
**Answer:** A CNN is a class of deep neural networks designed specifically for processing structured grid data like images. It utilizes convolutional layers with learnable filters (kernels) to extract spatial features like edges, textures, shapes, and complex object parts, while preserving spatial relationships between pixels.

#### Q8. What does YOLO stand for, and why is it called "You Only Look Once"?
**Answer:** YOLO stands for **You Only Look Once**. Unlike traditional two-stage object detectors (such as Faster R-CNN) that first generate potential region proposals and then run classification on each proposal separately, YOLO frames object detection as a single regression problem. It processes the entire image in a single neural network forward pass, predicting bounding boxes and class probabilities simultaneously, making it exceptionally fast.

#### Q9. Which version of YOLO is used in this project, and why?
**Answer:** This project uses **YOLOv8n** (Nano version) developed by Ultralytics. The nano variant has approximately 3.2 million parameters and a compact file size (~6.2 MB). It is selected because it runs fast on consumer CPU hardware (15–25 FPS) without requiring a dedicated high-end GPU, making it ideal for college demonstrations.

#### Q10. What classes are filtered in this project from the COCO dataset?
**Answer:** YOLO is pre-trained on the MS COCO dataset (80 categories). In this project, we filter specifically for traffic vehicle classes:
* Class 2: `car`
* Class 3: `motorcycle`
* Class 5: `bus`
* Class 7: `truck`

---

### Section 4: Accident Classification & Collision Logic

#### Q11. Why do we separate accident detection from vehicle detection?
**Answer:** Simply detecting vehicles does **not** mean an accident has occurred; vehicles are naturally present in every road traffic feed. Decoupling the pipeline allows YOLO to handle object localization, while a dedicated accident engine analyzes spatial interaction, relative distance, and motion anomalies to determine if a collision occurred.

#### Q12. What is Intersection over Union (IoU) and how is it used for collision detection?
**Answer:** Intersection over Union (IoU) is a geometric metric measuring the overlap between two bounding boxes:
$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
An IoU of 0 means vehicles are completely separate. When two vehicles collide, their 2D bounding boxes penetrate each other, resulting in a high IoU (typically $> 0.18$).

#### Q13. How does the system prevent false alarms caused by normal overtaking?
**Answer:** During normal lane overtaking, vehicles may temporarily appear close or slightly overlap for 1 or 2 frames due to perspective, but they quickly separate and maintain speed. Our system implements a **multi-frame persistence filter** requiring overlap to be sustained for $\ge 4$ consecutive frames, combined with a **kinematic velocity check** (sudden deceleration $> 40\%$), filtering out transient overtaking maneuvers.

#### Q14. What is the Overlap Ratio and why is it needed in addition to IoU?
**Answer:** When a very small vehicle (such as a motorcycle) collides with a massive vehicle (such as a multi-axle truck), the standard IoU can remain mathematically small because the denominator (union area) is dominated by the truck. The Overlap Ratio computes intersection relative to the smaller bounding box:
$$\text{Overlap Ratio} = \frac{\text{Intersection Area}}{\min(\text{Area}_A, \text{Area}_B)}$$
This ensures collisions involving small two-wheelers are reliably flagged.

---

### Section 5: Evaluation, Datasets, and System Limitations

#### Q15. What are False Positives and False Negatives in this project?
**Answer:**
* **False Positive (Type I Error):** The system triggers an accident alert when no accident actually happened (e.g., dense stopped traffic at a red light where vehicles are bumper-to-bumper).
* **False Negative (Type II Error):** An actual collision occurs, but the system fails to alert (e.g., a vehicle strikes a pedestrian or pole, which is not tracked as a pairwise vehicle collision).

#### Q16. What evaluation metrics are used to measure object detection performance?
**Answer:**
1. **Precision:** Ratio of correctly predicted vehicles to all predicted vehicles ($\frac{TP}{TP + FP}$).
2. **Recall:** Ratio of correctly detected vehicles to all actual vehicles on the road ($\frac{TP}{TP + FN}$).
3. **mAP (mean Average Precision):** The mean area under the Precision-Recall curve across all evaluated IoU thresholds and categories.

#### Q17. Does the system require an active internet connection to run?
**Answer:** An internet connection is required only once during the initial run to download the lightweight `yolov8n.pt` weights file (~6.2 MB). After the weights are saved locally in the `models/` folder, the entire video processing, vehicle detection, and collision analysis engine runs completely offline.

#### Q18. What are the key limitations of this prototype?
**Answer:**
1. **2D Monocular Occlusion:** A single camera flattens 3D space into 2D; two cars at different depths may visually overlap from the camera's angle.
2. **Weather and Lighting Sensitivity:** Extreme rain, dense fog, or night-time headlight glare can lower YOLO detection confidence.
3. **Single-Vehicle Crashes:** The pairwise collision heuristic focuses on vehicle-to-vehicle accidents; single-vehicle rollovers or tree collisions require dedicated optical flow or crash-classifier models.
4. **Human Verification Requirement:** Predictions are probabilistic aids and require human operator validation before dispatching emergency services.
