# `quadtank-fdi`: First-Principles Nonlinear Digital Twin & Observer-Based Fault Detection and Isolation (FDI)

[CI](https://www.google.com/search?q=https://github.com/sismax01/quadtank-fdi/actions/workflows/ci.yml)





A production-grade Python framework implementing a continuous-time nonlinear **Digital Twin**, **Multivariable Linear Quadratic Integral (LQI) Control**, **Continuous-Discrete Extended Kalman Filtering (CD-EKF)**, and **Structured Residual Fault Detection & Isolation (FDI)** for the benchmark **Johansson (2000) Quadruple-Tank Process**.

Built entirely from first principles—Bernoulli orifice differential equations, analytical Jacobian linearization, Continuous-Time Algebraic Riccati Equations (CARE), Joseph-stabilized Kalman filtering, and Analytical Redundancy Relations (ARRs)—for multivariable process control and industrial fault diagnosis.



## Key Engineering Highlights

- **Nonlinear Multivariable Plant Digital Twin (`plant.py`)**: Exact continuous-time 4-state nonlinear ODE model dynamic simulation using SciPy `solve_ivp` (Dormand-Prince RK45).
- **Closed-Loop MIMO LQI Controller (`controller.py`)**: Solves the 6-state Continuous-Time Algebraic Riccati Equation (CARE) for optimal tracking of coupled upper/lower tank setpoints.
- **Realistic Physical & Instrumentation Fault Engine (`faults.py`)**: Injects incipient sensor ramp drift (`SENSOR_DRIFT`), actuator efficiency loss (`ACTUATOR_EFFICIENCY_LOSS`), leak orifice discharge (`TANK_LEAK`), line blockages (`LINE_BLOCKAGE`), and process disturbance steps (`PROCESS_DISTURBANCE`).
- **CD-EKF & Structured Observer Bank (`observers.py`)**: Combines a Joseph-stabilized Continuous-Discrete Extended Kalman Filter for state estimation with a bank of dedicated observers for analytical redundancy.
- **Sequential CUSUM / GLRT Detection & Directed-Graph Isolation (`fdi.py`)**: Isolates all 6 benchmark operating faults using residual directional vector matching and cumulative sum hypothesis testing.



## 1. Benchmark Plant Physics: The Quadruple-Tank Process

The Quadruple-Tank Process (K. H. Johansson, *IEEE Trans. Control Systems Technology*, vol. 8, no. 3, pp. 456–465, May 2000) consists of four interconnected water tanks and two pumps.

- **Pump 1** delivers flow $\gamma_1 k_1 u_1$ directly into **Lower Tank 1** and $(1 - \gamma_1) k_1 u_1$ into **Upper Tank 4**.
- **Pump 2** delivers flow $\gamma_2 k_2 u_2$ directly into **Lower Tank 2** and $(1 - \gamma_2) k_2 u_2$ into **Upper Tank 3**.
- **Tanks 3 & 4** drain by gravity into **Lower Tanks 1 & 2** respectively through discharge orifices $a_3, a_4$.
- **Tanks 1 & 2** drain out of the plant through bottom discharge orifices $a_1, a_2$.

### Continuous-Time State Space Equations

Let $x(t) = [h_1(t), h_2(t), h_3(t), h_4(t)]^T \in \mathbb{R}_+^4$ represent the liquid levels in centimeters, and $u(t) = [u_1(t), u_2(t)]^T \in [0, 10]^2\text{ Volts}$ represent pump drive signals:

$$\frac{dh_1}{dt} = -\frac{a_1}{A_1}\sqrt{2g h_1} + \frac{a_3}{A_1}\sqrt{2g h_3} + \frac{\gamma_1 k_1}{A_1} u_1$$

$$\frac{dh_2}{dt} = -\frac{a_2}{A_2}\sqrt{2g h_2} + \frac{a_4}{A_2}\sqrt{2g h_4} + \frac{\gamma_2 k_2}{A_2} u_2$$

$$\frac{dh_3}{dt} = -\frac{a_3}{A_3}\sqrt{2g h_3} + \frac{(1 - \gamma_2) k_2}{A_3} u_2$$

$$\frac{dh_4}{dt} = -\frac{a_4}{A_4}\sqrt{2g h_4} + \frac{(1 - \gamma_1) k_1}{A_4} u_1$$

### Minimum-Phase vs. Non-Minimum Phase Operating Dynamics

The steady-state flow distribution is strictly governed by the split ratios $\gamma_1, \gamma_2 \in (0, 1)$:

- **Minimum-Phase ($\gamma_1 + \gamma_2 > 1$, e.g., $\gamma_1=0.70, \gamma_2=0.60$)**: Mass flow predominantly enters lower tanks directly. Transmission zeros of the linearized multivariable system lie strictly in the left half-plane (LHP).
- **Non-Minimum Phase ($\gamma_1 + \gamma_2 < 1$, e.g., $\gamma_1=0.30, \gamma_2=0.25$)**: Flow predominantly routes through upper tanks, causing a Right Half-Plane (RHP) transmission zero. Step changes produce inverse response (undershoot) on lower tank levels.

## 2. Linear Quadratic Integral (LQI) Control Architecture

To eliminate steady-state error under continuous process disturbance step loads, the plant state $x \in \mathbb{R}^4$ is augmented with integral states $q(t) = \int_0^t (r(\tau) - y(\tau))\,d\tau \in \mathbb{R}^2$ for the regulated outputs $y = [h_1, h_2]^T$:

$$\dot{z}(t) = \begin{bmatrix} \dot{x} \\ \dot{q} \end{bmatrix} = \begin{bmatrix} A & 0_{4\times 2} \\ -C & 0_{2\times 2} \end{bmatrix} z(t) + \begin{bmatrix} B \\ 0_{2\times 2} \end{bmatrix} u(t) + \begin{bmatrix} 0_{4\times 2} \\ I_{2\times 2} \end{bmatrix} r(t)$$

The Continuous-Time Algebraic Riccati Equation (CARE):

$$A_a^T P + P A_a - P B_a R^{-1} B_a^T P + Q_a = 0$$

yields the optimal full-state feedback gain matrix $K = R^{-1} B_a^T P = [K_x \;\vert\; K_q] \in \mathbb{R}^{2 \times 6}$. The control law becomes:

$$u(t) = \operatorname{sat}_{[0, 10]}\left( u_0 - K_x (x(t) - x_0) - K_q \int_0^t (r(\tau) - y(\tau))\,d\tau \right)$$

## 3. Continuous-Discrete Extended Kalman Filter (CD-EKF)

State estimation under measurement noise $v_k \sim \mathcal{N}(0, R_k)$ and process noise $w(t) \sim \mathcal{N}(0, Q_c)$ is performed via a Continuous-Discrete EKF:

**1. Continuous-Time State & Covariance Propagation ($t_k \to t_{k+1}$):**

$$\dot{\hat{x}}(t) = f(\hat{x}(t), u(t)), \qquad \dot{P}(t) = F(\hat{x}(t)) P(t) + P(t) F(\hat{x}(t))^T + Q_c$$

**2. Discrete-Time Innovation & Measurement Update:**

$$r_k = y_k - g(\hat{x}_{k\vert k-1}), \qquad H_k = \left. \frac{\partial g}{\partial x} \right|_{\hat{x}_{k\vert k-1}}, \qquad S_k = H_k P_{k\vert k-1} H_k^T + R_k$$

$$K_k = P_{k\vert k-1} H_k^T S_k^{-1}, \qquad \hat{x}_{k\vert k} = \hat{x}_{k\vert k-1} + K_k r_k$$

$$P_{k\vert k} = (I - K_k H_k) P_{k\vert k-1} (I - K_k H_k)^T + K_k R_k K_k^T \quad \text{(Joseph Form Stabilization)}$$

## 4. Analytical Redundancy & Fault Isolation Logic

Fault detection utilizes a bank of sub-state observers and analytical redundancy relations (ARRs) generating residual vectors $e_k = y_k - C \hat{x}_k$:

| Fault Type | ARR Residual Signature ($e_1, e_2, e_3, e_4$) | EKF Innovation Variance Ratio | Isolation Criterion |
| :--- | :--- | :--- | :--- |
| **Nominal** | $0,\; 0,\; 0,\; 0$ | $\approx 1$ | All residuals within $\pm 3\sigma_v$; $\text{NIS}_k < \chi^2_4$ |
| **Sensor $i$ Frozen** | $\delta_{i1},\; \delta_{i2},\; \delta_{i3},\; \delta_{i4}$ | $0$ on Ch $i$ | Residual variance drops to zero; sliding window variance ratio test drops below threshold |
| **Sensor $i$ Step Bias** | $\delta_{i1},\; \delta_{i2},\; \delta_{i3},\; \delta_{i4}$ | Spike | Single-sample CUSUM jump on $e_i$ matching signed direction |
| **Sensor $i$ Ramp Drift** | $\delta_{i1},\; \delta_{i2},\; \delta_{i3},\; \delta_{i4}$ | $\approx 1$ | Sustained linear accumulation in directional CUSUM statistic $S_{k,i}$ |
| **Pump 1 Degradation** | $< 0\text{ (primary)},\; < 0\text{ (cascade)},\; 0,\; < 0\text{ (primary)}$ | $\approx 1$ | Coupled negative directional signature on $e_1$ and $e_4$ |
| **Pump 2 Degradation** | $< 0\text{ (cascade)},\; < 0\text{ (primary)},\; < 0\text{ (primary)},\; 0$ | $\approx 1$ | Coupled negative directional signature on $e_2$ and $e_3$ |
| **Upper Tank 3 Leak** | $< 0\text{ (primary)},\; 0,\; < 0\text{ (primary)},\; 0$ | $\approx 1$ | Gravity cascade drop in **Tank 3** causing secondary depletion in **Tank 1** |
| **Lower Tank 1 Leak** | $< 0\text{ (primary)},\; 0,\; 0,\; 0$ | $\approx 1$ | Single lower tank; negative orifice transform residual |

## 5. Verification & Benchmark Test Suite Results

The full analytical test suite validates dynamic consistency, controller setpoint tracking, estimator variance bounds, and FDI classification performance across $100\%$ of benchmark fault scenarios:

| Scenario | Injected Fault & Channel | Detection Latency | Isolated Root Cause | Pre-Fault EKF RMSE | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `nominal` | Fault-free (+setpoint step at $t=25\text{ s}$) | None ($0\text{ FA}$) | `nominal` (None) | $0.012\text{ cm}$ | PASSED |
| `sensor_drift` | Sensor 3 ramp drift ($+0.025\text{ cm/s}$) | $7.0\text{ s}$ | `sensor_drift` (Ch 3) | $0.012\text{ cm}$ | PASSED |
| `sensor_frozen` | Sensor 2 stuck transmitter ($y_2(t) = y_2(100\text{ s})$) | $9.5\text{ s}$ | `sensor_frozen` (Ch 2) | $0.012\text{ cm}$ | PASSED |
| `sensor_bias` | Sensor 1 step bias ($+2.00\text{ cm}$) | $0.0\text{ s}$ | `sensor_bias` (Ch 1) | $0.0100\text{ cm}$ | PASSED |
| `actuator_degradation` | Pump 1 loss of effectiveness ($-28\%$, $\tau=5\text{ s}$) | $7.5\text{ s}$ | `actuator_degradation` (Pump 1) | $0.012\text{ cm}$ | PASSED |
| `tank_leak` | Tank 1 orifice leak ($a_{\text{leak}} = 0.018\text{ cm}^2$) | $5.5\text{ s}$ | `tank_leak` (Ch 1) | $0.012\text{ cm}$ | PASSED |

### Diagnostic & Control Plots

[./docs/assets/control_theory_analysis.png]
[./docs/assets/quad_tank_fdi_diagnostics.png]

## 6. Quickstart & Usage

### Installation

```bash
git clone https://github.com/sismax01/quadtank-fdi.git
cd quadtank-fdi
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -e .
```

### Execution

Run the diagnosis benchmark harness to simulate all fault injection scenarios and generate diagnostic plots and CSV metrics:

```bash
make run
# Or invoke the CLI module directly:
python -m quadtank_fdi.cli
```

Run the automated `pytest` suite verifying dynamic bounds, observer convergence, and FDI metrics:

```bash
make test
# Or run pytest directly:
pytest tests/ -v
```

## License

Distributed under the MIT License. See `LICENSE` for details.
