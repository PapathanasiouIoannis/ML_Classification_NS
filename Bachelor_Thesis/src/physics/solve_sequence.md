# Stellar Sequence Solver (Mass-Radius Integration)

## Role in the Project
This script is the computational engine responsible for constructing the **Mass–Radius relation** for a given Equation of State (EoS). While the EoS defines the properties of matter (how pressure relates to density), this script calculates how that matter behaves under self-gravity to form a star.

It generates a complete "sequence" of stars—ranging from low-mass bodies to the maximum stable mass—by systematically varying the central pressure. For each central pressure, the Tolman–Oppenheimer–Volkoff (TOV) equations are integrated from the center to the surface to determine the total mass $M$, radius $R$, and tidal deformability $\Lambda$.

## Physics and Equations

### 1. The Central Boundary Condition
To solve the differential equations of stellar structure, the conditions at the center of the star ($r=0$) must be defined. A sequence of stars is parameterized by the **central pressure** $P_c$.

For each chosen $P_c$, the corresponding **central energy density** $\epsilon_c$ and **speed of sound** $c_s$ are calculated based on the specific EoS model.

#### Quark Matter (CFL Model)
For quark stars, the relationship between pressure and energy density is derived from the thermodynamic potential of the Color-Flavor-Locked phase. The pressure $P$ is related to the quark chemical potential $\mu$ by:

$$
P = \frac{3}{4\pi^2}\mu^4 + \frac{3}{\pi^2}\left(\Delta^2 - \frac{m_s^2}{4}\right)\mu^2 - B
$$

Here, $B$ is the Bag constant, $\Delta$ is the pairing gap, and $m_s$ is the strange quark mass. This quadratic equation is solved for $\mu^2$, which is then used to compute the energy density $\epsilon$.

#### Hadronic Matter (Scaled and Mixed)
For hadronic stars, the central conditions are determined by mixing two parent EoS models ($A$ and $B$) and applying a homologous scaling factor $\alpha$. If $P_{base} = P_c / \alpha$, the energy density is computed as:

$$
\epsilon_c = \alpha \left[ \epsilon_A(P_{base})^w \cdot \epsilon_B(P_{base})^{(1-w)} \right]
$$

where $w$ is the mixing weight. This allows the simulation to explore a continuous space of hadronic models between discrete analytical solutions.

### 2. Tidal Deformability and the Love Number
In addition to Mass and Radius, the script computes the **tidal deformability** $\Lambda$, which measures how easily the star is deformed by an external gravitational field (such as a companion in a binary merger).

The calculation requires the **Tidal Love Number** $k_2$, which is derived from the value of the tidal perturbation variable $y_R$ at the stellar surface. The Love number is given by a complex function of the compactness $C = GM/Rc^2$:

$$
k_2 = \frac{8C^5}{5} (1-2C)^2 \left[ 2 + 2C(y_R - 1) - y_R \right] \times \{ \dots \}^{-1}
$$

(The denominator $\{ \dots \}$ contains logarithmic terms ensuring the continuity of the metric potential).

The dimensionless tidal deformability is then:

$$
\Lambda = \frac{2}{3} k_2 C^{-5}
$$

### 3. Stability and Causality
*   **Stability:** The sequence is terminated when the maximum mass is reached ($dM/dP_c = 0$). Stars beyond this point are unstable to gravitational collapse.
*   **Causality:** The speed of sound is monitored at the center. If the EoS implies a sound speed greater than the speed of light ($c_s > 1$), the solution is flagged or clamped to physical limits.

## Algorithm and Calculations

1.  **Pressure Grid Generation**
    A logarithmic grid of central pressures is defined.
    *   For **Quark stars**, the grid starts at higher pressures to overcome the vacuum pressure $B$ (since $P=0$ at finite density).
    *   For **Hadronic stars**, the grid spans from the low-density crust regime to high-density core pressures.

2.  **Thermodynamic Initialization**
    For every pressure step in the grid:
    *   The EoS parameters are unpacked.
    *   The corresponding central energy density and speed of sound are computed using the physics described above.
    *   If the parameters yield an unphysical state (e.g., negative density), that pressure step is skipped.

3.  **Numerical Integration**
    The TOV equations are integrated radially outward from a small non-zero radius ($r \approx 10^{-4}$ km) to avoid the geometric singularity at the center.
    *   The integration proceeds until the pressure drops to zero (the surface).
    *   The solver returns the total Mass $M$, Radius $R$, and the surface value of the tidal variable $y_R$.

4.  **Observable Extraction**
    *   The Tidal Love number $k_2$ and Deformability $\Lambda$ are calculated from the surface values.
    *   Results are filtered to remove black holes (where $R \le 2GM/c^2$) or numerically failed integrations.
    *   The Maximum Mass found so far in the sequence is tracked.

## Inputs and Outputs

### Inputs
*   **EoS Parameters:** A tuple containing the specific model configuration.
    *   Quark: $(B, \Delta, m_s)$
    *   Hadronic: Functions for parent models, mixing weight $w$, scaling factor $\alpha$, and transition pressure.
*   **Star Type Flag:** A boolean indicating whether to use Quark (CFL) or Hadronic physics logic.

### Outputs
*   **Curve Data:** A list of vectors, where each vector represents one star in the sequence:
    $$ [Mass, Radius, \Lambda, P_c, \epsilon_c, c_{s,c}^2, \epsilon_{surface}] $$
*   **Maximum Mass:** The highest stable mass achieved by this EoS model.

## Connection to the Overall Workflow
This script is the **workhorse** of the simulation pipeline.
*   It is called repeatedly by `worker_hadronic_gen.py` and `worker_quark_gen.py`.
*   It relies on `tov_rhs.py` to provide the derivatives for the integrator.
*   The lists of stars produced here are aggregated to form the final `thesis_dataset.csv`, which provides the training data for the Machine Learning models.
