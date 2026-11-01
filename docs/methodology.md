# Research Methodology & Technical Formulation

## 1. Mathematical Formulation & Coordinate Frame

All physical simulation and trajectory forecasting are conducted in a **2D metric Cartesian coordinate frame** $(x, y) \in \mathbb{R}^2$ with standard SI units:
- **Positions:** $x, y$ in metres ($m$).
- **Velocities:** $v_x = \dot{x}, v_y = \dot{y}$ in metres per second ($\text{m/s}$).
- **Time step:** Uniform discretization $\Delta t = 0.1\text{ s}$ ($10\text{ Hz}$).
- **Observation Horizon:** $T_{obs} = 2.0\text{ s}$ (20 discrete time steps).
- **Prediction Horizon:** $T_{pred} = 3.0\text{ s}$ (30 discrete time steps).

### Relative Coordinate Normalization
To ensure translation invariance, trajectory prediction models operate relative to the observation endpoint $\vec{p}(t_{obs}) = (x_{obs}, y_{obs})$:
$$\Delta \vec{p}(\tau) = \vec{p}(t_{obs} + \tau) - \vec{p}(t_{obs}), \quad \tau \in \{1, \dots, T_{pred}\}$$

---

## 2. Trajectory Forecasting Models

### Model 1: Constant-Velocity (CV) Baseline
A zero-parameter, deterministic physics baseline assuming unaccelerated motion from the observation endpoint velocity vector $\vec{v}(t_{obs}) = (v_{x,obs}, v_{y,obs})$:
$$\hat{\Delta x}(k) = v_{x,obs} \cdot (k \cdot \Delta t), \quad \hat{\Delta y}(k) = v_{y,obs} \cdot (k \cdot \Delta t), \quad k \in \{1, \dots, T_{pred}\}$$

### Model 2: Deterministic Recurrent Forecaster (TrajectoryGRU)
A compact Gated Recurrent Unit sequence-to-sequence neural network:
- **Encoder:** 2-layer GRU with hidden dimension $h=64$, processing input feature sequence $\mathbf{X} \in \mathbb{R}^{T_{obs} \times 6}$ ($[\Delta x, \Delta y, v_x, v_y, \text{speed}, \theta]$).
- **Projection Head:** Fully connected layers mapping the final hidden state $h_{T_{obs}}$ to future displacements $\hat{\mathbf{Y}} \in \mathbb{R}^{T_{pred} \times 2}$.
- **Loss Function:** Mean Squared Error (MSE):
  $$\mathcal{L}_{MSE} = \frac{1}{T_{pred}} \sum_{k=1}^{T_{pred}} \|\Delta \vec{p}_k - \hat{\Delta \vec{p}}_k\|^2$$

### Model 3: Uncertainty-Aware Forecaster (Bivariate Gaussian GRU)
Predicts a full probabilistic spatial distribution over future positions $(\mu_x, \mu_y, \sigma_x, \sigma_y, \rho)_k$ for each future step $k$:
- **Bivariate Normal Distribution:**
  $$\mathcal{N}(\vec{\mu}_k, \boldsymbol{\Sigma}_k), \quad \boldsymbol{\Sigma}_k = \begin{bmatrix} \sigma_{x,k}^2 & \rho_k \sigma_{x,k} \sigma_{y,k} \\ \rho_k \sigma_{x,k} \sigma_{y,k} & \sigma_{y,k}^2 \end{bmatrix}$$
- **Loss Function:** Negative Log-Likelihood (NLL):
  $$\mathcal{L}_{NLL} = \frac{1}{2} \left[ \frac{z_{x,k}^2 - 2\rho_k z_{x,k} z_{y,k} + z_{y,k}^2}{1 - \rho_k^2} \right] + \ln \sigma_{x,k} + \ln \sigma_{y,k} + \frac{1}{2} \ln(1 - \rho_k^2) + \ln(2\pi)$$
  where $z_{x,k} = \frac{x_k - \mu_{x,k}}{\sigma_{x,k}}$ and $z_{y,k} = \frac{y_k - \mu_{y,k}}{\sigma_{y,k}}$.

---

## 3. Geometric Collision & Risk Mechanics

### Exact 2D Collision Detection: Separating Axis Theorem (SAT)
Each road user is modeled as an Oriented Bounding Box (OBB) defined by center $(x, y)$, length $L$, width $W$, and heading angle $\theta$.
Two oriented bounding boxes $B_1$ and $B_2$ are in physical collision if and only if their 1D projections overlap across all 4 normal candidate axes $\mathbf{u} \in \{\mathbf{u}_{1}^{\parallel}, \mathbf{u}_{1}^{\perp}, \mathbf{u}_{2}^{\parallel}, \mathbf{u}_{2}^{\perp}\}$.

### Metric Separation Distance
$$\text{dist}(B_1, B_2) = \begin{cases} 0.0 & \text{if SAT collision detected} \\ \min_{\mathbf{c}_1 \in B_1, \mathbf{c}_2 \in B_2} \|\mathbf{c}_1 - \mathbf{c}_2\| & \text{otherwise} \end{cases}$$

### Physical Time-To-Collision (TTC)
For vehicles with relative displacement $\vec{r} = \vec{p}_{act} - \vec{p}_{ego}$ and relative velocity $\vec{v}_{rel} = \vec{v}_{act} - \vec{v}_{ego}$, physical TTC is the smallest positive real root $t > 0$ satisfying:
$$\|\vec{r} + \vec{v}_{rel} \cdot t\|^2 \le R_{contact}^2$$

---

## 4. Safety Control Policies

1. **Reactive Baseline:**
   - Evaluates only immediate separation $d_{sep}$ and instantaneous TTC.
   - If $d_{sep} \le 2.0\text{m}$ or $\text{TTC} \le 1.8\text{s}$: Commands hard emergency braking ($a = -6.5\text{ m/s}^2$).
   - If $d_{sep} \le 5.0\text{m}$ or $\text{TTC} \le 3.2\text{s}$: Commands advisory braking ($a = -2.5\text{ m/s}^2$).
   - Vulnerability: Prone to late, violent emergency stops when obstacles suddenly enter the lane.

2. **Prediction-Aware Policy:**
   - Rolls forward predicted future trajectories $\hat{\mathbf{Y}}_{1:T_{pred}}$.
   - Computes minimum predicted separation $\min_k \text{dist}(B_{ego}(k), B_{act}(k))$.
   - Initiates proactive, smooth deceleration ($a \approx -3.0\text{ m/s}^2$) up to $2.5\text{s}$ before a conflict materializes.

3. **Uncertainty-Aware Policy:**
   - Evaluates the conservative $(1 - \alpha)$ confidence boundary of the predictive covariance ellipse directed towards the ego vehicle:
     $$d_{conservative}(k) = \|\vec{p}_{ego}(k) - \vec{\mu}_k\| - \sqrt{-2\ln(1 - \alpha)} \cdot \sigma_{dir}(k) - (R_{ego} + R_{act})$$
   - When motion is unpredictable (high entropy/variance), expands the clearance boundary smoothly. When prediction certainty is high, cruises efficiently without unnecessary braking.
