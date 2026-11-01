# Dataset Audit: ThirdEye Labs Indian Road Driving Dataset

**Repository ID:** `thirdeyelabs/indian-road-dataset`  
**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)  
**Access Tier:** Gated (Automated acceptance on Hugging Face)  
**Audit Date:** October 2026  
**Auditor:** Lead Research & Simulation Engineer  

---

## 1. Executive Summary & Project Role

The **Indian Road Driving Dataset** (`thirdeyelabs/indian-road-dataset`) is an open dataset created by ThirdEye Labs capturing dashcam footage across urban and suburban Delhi NCR, India.

In this research project, **Data Source A (Indian Road Dataset)** fulfills a specific, transparent role:
1. **Descriptive Class Taxonomy**: It defines realistic object categories unique to Indian road scenes (autorickshaws, unridden/ridden motorcycles, cattle/animals, pedestrians, light commercial vehicles).
2. **Behavioral Inspiration**: It grounds our synthetic simulation scenarios (e.g., informal lane filtering, motorcycle cut-ins, pedestrian jaywalking, mixed-traffic speed variance).
3. **Sensor & Viability Audit**: It provides empirical justification for why monocular dashcam bounding boxes cannot serve as uncalibrated ground-truth metric coordinates for physical trajectory forecasting and physical collision risk.

---

## 2. Dataset Architecture & Storage Statistics

| Parameter | Documented Value | Verified in Audit |
| :--- | :--- | :--- |
| **Total Storage** | ~460.9 GB (429.3 GiB) | Verified (`usedStorage: 460954268309` bytes) |
| **Archive Structure** | 646 WebDataset tar shards | Verified (`data/train-00000-of-00646.tar` to `00645`) |
| **Total Video Clips** | 8,437 clips | Verified from dataset card |
| **Annotated Keyframes**| 645,714 frames | Verified |
| **Object Detections** | 3,669,119 bounding boxes | Verified |
| **Segmentation Masks**| 1,290,463 masks | Verified |
| **Keyframe Sampling** | 1.0 frame per second (1.0 Hz) | Verified (FFmpeg keyframe extraction) |
| **Per-Clip GPS** | ~66% of clips with GPS tracks | Verified (`gps/gps_tracks.json`) |

---

## 3. Label Provenance & Annotation Quality

Labels are **machine-generated pseudo-labels**, not gold human annotations:
- **Detection Model:** YOLO11m fine-tuned on the India Driving Dataset (IDD).
- **Tracking Algorithm:** ByteTrack (2D bounding box association across frames).
- **Segmentation Model:** SegFormer-B5 fine-tuned on IDD.
- **Scene Attributes:** CLIP ViT-L/14.
- **BDD100K Flag:** Every bounding box explicitly specifies `manualShape: false` (machine annotation flag in BDD100K schema).
- **Confidence Profile:** Approximately **33% of all detections fall below 0.5 confidence**, exhibiting significant pseudo-label noise.

---

## 4. Object Class Taxonomy (12 Classes)

The dataset defines 12 object classes, including 3 classes with no equivalent in standard Western benchmarks (BDD100K, nuScenes, Waymo):

1. `person`: Pedestrians walking, standing, or crossing.
2. `rider`: Motorcyclists, scooterists, and cyclists with rider.
3. `car`: Passenger cars, taxis, hatchbacks, sedans.
4. `truck`: Heavy transport trucks, dumpers, and light commercial tempos.
5. `bus`: State transit, private charter, and school buses.
6. `motorcycle`: Parked / unridden motorcycles and scooters.
7. `bicycle`: Unridden pedal bicycles.
8. `autorickshaw`: Three-wheeled motorized passenger vehicles (tuk-tuks). *(Unique Indian road class)*
9. `animal`: Cattle, stray dogs, and horses on the carriageway. *(Unique Indian road class)*
10. `vehicle fallback`: Informal vehicles (e.g. cycle-rickshaws, handcarts, tractor-trailers). *(Unique Indian road class)*
11. `traffic light`: Signal lamps.
12. `traffic sign`: Regulatory and advisory boards.

---

## 5. Technical Assessment: Why Pixel Tracks Cannot Be Used for Metric Trajectory Forecasting

Physical trajectory forecasting and collision avoidance require metric (x, y) coordinates with dense temporal resolution. The audit confirms the following hard technical constraints:

1. **Uncalibrated Monocular Projection:**
   Bounding boxes $[x_1, y_1, x_2, y_2]$ are given in pixel units. The recording hardware is consumer CP Plus dashcams with unknown focal length $f$, unknown camera mounting height $h$, and variable pitch/roll angles due to vehicle suspension bounce. Pixels cannot be inverted to metric ground positions $(X_m, Y_m)$ without a known extrinsic matrix and flat-road plane homography $H$.

2. **Temporal Sparsity (1.0 Hz Sampling):**
   Keyframes are extracted at 1 frame per second ($1.0\text{ Hz}$). Vehicle dynamics, emergency braking, and evasive steering occur on time scales of $0.1\text{ s}$ to $0.5\text{ s}$ ($10\text{ Hz}$). An auto-rickshaw or motorcycle moving at $36\text{ km/h}$ travels $10\text{ metres}$ between successive frames, rendering fine-grained cut-in trajectory modeling impossible.

3. **Tracking Continuity & ID Switches:**
   ByteTrack operating on $1\text{ Hz}$ monocular frames experiences frequent ID switches during dense occlusions, yielding short, fragmented tracks without physical velocity vectors.

4. **Invalidity of Pixel-Based Time-To-Collision (TTC):**
   Physical $\text{TTC} = \frac{d_{rel}}{v_{rel}}$ in metres and $\text{m/s}$. Approximating TTC using pixel expansion rates $(\frac{w}{\dot{w}})$ assumes constant camera motion and optical flow symmetry, which fails catastrophically under Indian road conditions (pothole bounce, steering yaw, uncoordinated multi-agent movement).

---

## 6. Conclusion & Project Strategy

In accordance with scientific rigor:
- We **do not fabricate** false metric conversions from uncalibrated pixel bounding boxes.
- We utilize the verified 12-class taxonomy, vehicle dimensions, and scenario archetypes to parameterize our **metric 2D synthetic traffic simulator (10 Hz, SI metres, explicit bounding box geometries)**.
- Trajectory prediction models (Constant Velocity, GRU/LSTM) and collision-risk algorithms are trained and benchmarked within reproducible metric environments where ground-truth safety outcomes can be evaluated without ambiguity.
