# TOV and Tidal Evolution Kernel

## Role in the Project
This script defines the system of coupled ordinary differential equations (ODEs) that govern the internal structure of a static, spherically symmetric neutron star. It serves as the "derivative function" (or Right-Hand Side kernel) required by the numerical integrator.

For every step in the radial integration, this module computes the gradients of mass, pressure, and tidal deformability. It acts as the bridge between General Relativity (macrophysics) and the Equation of State (microphysics), translating a local pressure into an energy density and sound speed, which then dictate the curvature of spacetime and the star's response to tidal fields.

## Physics and Equations

### 1. Tolman–Oppenheimer–Volkoff (TOV) Equations
The hydrostatic equilibrium of the star is described by the TOV equations, derived from Einstein's Field Equations for a perfect fluid.

The gradient of pressure $P$ with respect to radius $r$ is given by:

$$
\frac{dP}{dr} = - \frac{G \epsilon(r) m(r)}{r^2} \left(1 + \frac{P(r)}{\epsilon(r)}\right) \left(1 + \frac{4\pi r^3 P(r)}{m(r)}\right) \left(1 - \frac{2Gm(r)}{c^2 r}\right)^{-1}
$$

The gradient of the enclosed mass $m(r)$ is:

$$
\frac{dm}{dr} = 4\pi r^2 \epsilon(r)
$$

Here:
*   $P(r)$ is the local pressure.
*   $\epsilon(r)$ is the local energy density.
*   $m(r)$ is the gravitational mass enclosed within radius $r$.
*   $G$ and $c$ are the gravitational constant and speed of light (handled via conversion factors in the computation).

### 2. Tidal Deformability (Riccati Equation)
To determine the tidal deformability $\Lambda$ (a key observable for gravitational wave astronomy), a perturbation to the metric is evolved alongside the structure. This is formulated as a Riccati equation for the variable $y(r)$:

$$
\frac{dy}{dr} = - \frac{1}{r} \left( y^2 + y F(r) + r^2 Q(r) \right)
$$

The terms $F(r)$ and $Q(r)$ encapsulate the metric potentials and the stiffness of the matter. Notably, $Q(r)$ depends inversely on the speed of sound squared, $c_s^2$:

$$
Q(r) \propto \frac{5\epsilon + 9P + (\epsilon + P)/c_s^2}{1 - 2Gm/r} - \dots
$$

This dependence highlights why the speed of sound is critical: a "stiffer" EoS (higher $c_s^2$) resists tidal deformation more strongly.

### 3. Microphysics and Thermodynamics
The local energy density $\epsilon$ and sound speed $c_s^2$ are derived from the pressure $P$ dynamically, depending on the matter type.

**For Quark Matter (CFL Model):**
The thermodynamics are governed by the bag constant $B$, the pairing gap $\Delta$, and the strange quark mass $m_s$. The relationship is derived by solving for the quark chemical potential $\mu$. The pressure equation is inverted algebraically:

$$
P = \frac{3}{4\pi^2}\mu^4 + \frac{3}{\pi^2}\left(\Delta^2 - \frac{m_s^2}{4}\right)\mu^2 - B
$$

Once $\mu$ is found, the energy density is computed as:

$$
\epsilon = \Omega(\mu) + 2B \quad \text{(Conceptually)}
$$

**For Hadronic Matter:**
The energy density is obtained via interpolation or analytic functions describing the crust (SLy model) and the core. In the core, mixing between two parent models is applied using a scaling factor $\alpha$ and a mixing weight $w$.

### 4. Causality and Stability Constraints
Physical consistency is enforced locally at every integration step:
*   **Causality:** The speed of sound is clamped such that $c_s^2 \le 1$ (units of $c^2$). If the EoS implies a sound speed greater than light, it is artificially capped to preserve causality.
*   **Stability:** A minimum floor for $c_s^2$ is enforced to prevent numerical instabilities where $dP/d\epsilon \approx 0$.

## Algorithm and Calculations

1.  **State Unpacking:**
    The integration state vector $[m, P, y]$ is unpacked at the current radius $r$.

2.  **Thermodynamic Inversion:**
    *   The pressure $P$ is converted to geometric units.
    *   If the star is **Quark-type**, a quadratic equation is solved to find the chemical potential $\mu^2$ corresponding to $P$. This allows the calculation of $\epsilon$ and the analytic derivative $d\epsilon/dP$ (inverse sound speed).
    *   If the star is **Hadronic**, the pressure is checked against transition thresholds. If $P$ is in the crust, crustal functions are used. If in the core, the mixing formula is applied to combined parent EoS functions.

3.  **Constraint Application:**
    The calculated speed of sound is checked against the causal limit ($c_s \le 1$). If the microphysics predicts superluminal sound, the value is clamped to 1.0.

4.  **Metric Calculation:**
    The metric factor $(1 - 2Gm/r)^{-1}$ is computed. If the star approaches the Schwarzschild limit (forming a black hole), the derivative calculation is aborted.

5.  **Gradient Computation:**
    The values for $dm/dr$, $dP/dr$, and $dy/dr$ are calculated using the equations above and returned to the solver.

## Inputs and Outputs

### Inputs
*   **Radius ($r$):** The current radial position in the star.
*   **State Vector ($y_{state}$):** A vector containing $[Mass, Pressure, Tidal\_Variable]$.
*   **EoS Data:**
    *   For Quark: A tuple $(B, \Delta, m_s)$.
    *   For Hadronic: A tuple containing analytic functions for the two parent models, the crust model, and mixing parameters.

### Outputs
*   **Derivatives:** A list $[dm/dr, dP/dr, dy/dr]$ describing how the star's properties change with a small step in radius.

## Connection to the Overall Workflow
This function is not called directly by the main script but is passed as a callback to the ODE solver (typically `scipy.integrate.solve_ivp`) within the `solve_sequence` module. It is the fundamental physics kernel that allows the conversion of an Equation of State (pressure vs. density) into a macroscopic Observable (Mass vs. Radius vs. Tidal Deformability).
