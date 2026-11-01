# Research Limitations, Assumptions, and Safety Disclaimers

## 1. Scope & Research Boundary Disclaimers

- **Not an Autonomous Vehicle Driver:** This software repository is a **research simulation prototype**, not a production autonomous driving stack, vehicle controller, or safety-critical driver assistance system.
- **No Human Intention Understanding:** Trajectory models extrapolate kinematic patterns and spatial correlations. The models do not possess cognitive awareness of human intentions, eye contact, gestures, or traffic psychology.
- **Zero Real-World Deployment Readiness:** Under no circumstances should this code be deployed on real vehicles, physical robots, or embedded drive-by-wire hardware.

---

## 2. "Indian-Road-Inspired" vs. Real-World Validity

- **Inspiration, Not Ground Truth:** The term *"Indian-road-inspired"* describes our deliberate selection of multi-agent scenario archetypes (informal lane filtering, aggressive two-wheeler cut-ins, pedestrian mid-street hesitation, and three-wheeled autorickshaw speed variance).
- **Synthetic Simplifications:** Simulated actors adhere to parameterized kinematic trajectories and simplified behavioral profiles. Real Indian road environments feature complex road surface irregularities (potholes, unpaved shoulders), informal right-of-way negotiations, horn audio cues, and multi-modal weather dynamics that are not captured in a 2D planar kinematic simulation.

---

## 3. Data Source Limitations (Hugging Face Indian Road Dataset)

1. **Uncalibrated Monocular Projection:**
   The `thirdeyelabs/indian-road-dataset` provides 2D image bounding boxes $[x_1, y_1, x_2, y_2]$ from consumer CP Plus dashcams. Without camera intrinsic calibration matrices ($K$), known camera mounting heights, and roll/pitch angles, pixel bounding boxes cannot be inverted to ground-plane metric coordinates (metres) without severe distortion.
2. **Temporal Resolution (1.0 Hz):**
   Dashcam keyframes are extracted at $1\text{ frame per second}$. Physical emergency braking and collision avoidance occur on time scales of $0.1\text{ s}$ to $0.3\text{ s}$. Modeling high-speed trajectory dynamics at 1 Hz introduces excessive discretization error.
3. **Machine-Generated Pseudo-Labels:**
   The dataset annotations are produced automatically by YOLO11m, ByteTrack, and SegFormer models (`manualShape: false`). Approximately $33\%$ of detections fall below $0.5$ confidence, containing label noise, false positives, and tracking identity switches.

---

## 4. Modeling & Simulation Assumptions

- **Planar Road Surface:** Road gradient, superelevation, and uneven pavement friction are assumed to be zero (flat 2D plane).
- **Constant Friction Coefficient:** Tire-road surface friction is assumed constant with fixed maximum deceleration bounds ($-6.5\text{ m/s}^2$ emergency, $-2.5\text{ m/s}^2$ comfort).
- **Non-Reactive Surrounding Actors:** Surrounding actors follow parameterized scenario trajectories and do not steer away in response to the ego vehicle's braking actions. This provides a strict, reproducible safety stress-test.
- **Interpretation of Zero Collisions:** Observing zero collisions across evaluated test episodes does **NOT** constitute proof of zero operational risk. Rare, out-of-distribution traffic interactions could lead to failure modes not captured in the evaluated scenario seeds.
