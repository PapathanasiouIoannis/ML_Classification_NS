# Physical Constants and Simulation Configuration

## Role in the Project
This script serves as the central repository for all physical constants, unit conversions, and model constraints used throughout the research project. It defines the "laws of physics" for the simulation, ensuring that all modules—from the Equation of State (EoS) generators to the Tolman–Oppenheimer–Volkoff (TOV) solvers—operate within a consistent numerical and physical framework.

It establishes the boundaries of the parameter space for both Hadronic and Quark matter models, setting the prior distributions from which the synthetic neutron star population is sampled.

## Physics and Equations

### Fundamental Units and Conversions
The simulation operates by bridging the microscopic world of nuclear physics with the macroscopic world of general relativity.

*   **Nuclear Scale:** The fundamental unit of conversion between length and energy is the reduced Planck constant times the speed of light, $\hbar c$.
    $$
    \hbar c \approx 197.33 \text{ MeV} \cdot \text{fm}
    $$
    This constant allows quantities to be converted between inverse fermis ($\text{fm}^{-1}$) and energy ($\text{MeV}$).

*   **Gravitational Coupling:** To solve the TOV equations, microscopic energy densities (in $\text{MeV}/\text{fm}^3$) must be coupled to macroscopic geometry (in $\text{km}^{-2}$). A conversion factor $G_{conv}$ is derived from the gravitational constant $G$ and the speed of light $c$:
    $$
    G_{conv} \propto \frac{G}{c^4}
    $$
    This factor scales the energy density $\epsilon$ and pressure $P$ in the stress-energy tensor $T_{\mu\nu}$ to the curvature source in the Einstein Field Equations ($G_{\mu\nu} = 8\pi G T_{\mu\nu}$).

*   **Geometric Mass:** Stellar masses are often computed in geometric length units ($\text{km}$) before being converted to solar masses ($M_{\odot}$). The conversion factor $A_{conv}$ relates these units:
    $$
    M_{\text{geo}} = \frac{G M_{\text{phys}}}{c^2}
    $$

### Crust Physics (SLy Model)
For Hadronic stars, the Equation of State at low densities (the crust) is modeled using the SLy (Skyrme-Lyon) effective interaction. The crust is divided into regions separated by transition pressures $P_{trans}$. These thresholds define where the physics changes, such as the transition from the outer crust (nuclei in a lattice) to the inner crust (neutron drip).

### Quark Model Parameter Space (CFL)
The parameters defining the Color-Flavor-Locked (CFL) quark matter phase are constrained to physically motivated ranges:

1.  **Bag Constant ($B$):** Represents the vacuum pressure difference between the perturbative vacuum and the true vacuum. It confines quarks within the star.
    $$
    B \in [57.0, 400.0] \text{ MeV}/\text{fm}^3
    $$
    The lower bound ensures stability relative to ordinary nuclear matter (iron).

2.  **Pairing Gap ($\Delta$):** Represents the energy gap formed by Cooper pairing of quarks.
    $$
    \Delta \in [57.0, 250.0] \text{ MeV}
    $$
    A larger gap significantly stiffens the Equation of State, allowing for more massive quark stars.

3.  **Strange Quark Mass ($m_s$):** The effective mass of the strange quark.
    $$
    m_s \in [80.0, 120.0] \text{ MeV}
    $$
    This parameter introduces flavor symmetry breaking in the quark phase.

### Numerical Constraints
To ensure numerical stability during the integration of the differential equations:
*   **Radial Bounds:** The integration starts at a small non-zero radius $R_{min}$ (to avoid the singularity at $r=0$) and stops at a maximum radius $R_{max}$.
*   **Pressure Floor:** A safe minimum pressure is defined to detect the surface of the star ($P \to 0$).

## Algorithm and Calculations
While this script does not contain active algorithms, it defines the **sampling strategy** for the Monte Carlo generation:

1.  **Parameter Sampling:** The ranges defined here (e.g., `Q_B_MIN`, `Q_DELTA_RANGE`) are used by the worker scripts to draw random samples from a uniform prior distribution.
2.  **Causality Enforcement:** A threshold for energy density is set (`CAUSALITY_EPS_LIMIT`). If the speed of sound squared $c_s^2 = dP/d\epsilon$ exceeds the speed of light ($c_s^2 > 1$) below this density, the model is flagged as unphysical and discarded.
3.  **Buchdahl Limit:** A factor is defined to represent the general relativistic limit for the compactness of a static sphere:
    $$
    R \ge \frac{9}{4} \frac{GM}{c^2} \approx 2.25 \frac{GM}{c^2}
    $$
    This is used in visualization to shade the "forbidden" region of the Mass-Radius diagram.

## Inputs and Outputs
*   **Inputs:** This file requires no external inputs; it represents the foundational axioms of the simulation.
*   **Outputs:** It provides a structured dictionary (`CONSTANTS`) containing all physical parameters, unit conversions, plotting limits, and the data schema (column names for the dataset).

## Connection to the Overall Workflow
This configuration file is imported by every major component of the pipeline:
*   **Physics Engine:** Uses the conversion factors and EoS parameter ranges to generate stellar models.
*   **Machine Learning:** Uses the column schema to organize features (Mass, Radius, Lambda) and labels.
*   **Visualization:** Uses the plotting limits to ensure all figures share consistent axes and units.

Any change to the physical assumptions (e.g., modifying the range of the Bag constant or the resolution of the solver) is made centrally in this file.
