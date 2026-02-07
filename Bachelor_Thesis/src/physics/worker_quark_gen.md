# Quark Matter Equation of State Generation (CFL Model)

## Role in the Project
This script is responsible for generating the **Quark Star** portion of the synthetic training dataset (Label = 1). It implements the Color-Flavor-Locked (CFL) equation of state, which describes superfluid quark matter at extreme densities.

Unlike the hadronic generation process which relies on mixing existing models, this script relies on a parametric approach derived from Quantum Chromodynamics (QCD). A specialized **Target-Driven Inverse Sampling** technique is applied to ensure the dataset covers the full range of physically possible masses, particularly the high-mass region ($M > 2.0 M_{\odot}$), which is critical for training robust machine learning classifiers.

## Physics and Equations

### 1. The CFL Equation of State
The thermodynamic behavior of the star is governed by the grand canonical potential $\Omega$ for the Color-Flavor-Locked phase. The pressure $P$ and energy density $\epsilon$ are determined by three microscopic parameters:
1.  **Bag Constant ($B$):** The vacuum energy density that confines quarks.
2.  **Pairing Gap ($\Delta$):** The energy gap arising from Cooper pairing of quarks.
3.  **Strange Quark Mass ($m_s$):** The effective mass of the strange quark.

The relationship between pressure and the quark chemical potential $\mu$ is given by:

$$
P(\mu) = \frac{3}{4\pi^2}\mu^4 + \frac{3}{\pi^2}\left(\Delta^2 - \frac{m_s^2}{4}\right)\mu^2 - B
$$

The energy density $\epsilon$ is derived thermodynamically from $\Omega = -P$.

### 2. The Stability Window
Not all combinations of ($B, \Delta, m_s$) produce stable quark matter. For a quark star to exist (and not decay into normal nuclear matter or a black hole), the energy per baryon at zero pressure must be lower than that of the neutron ($M_n \approx 939$ MeV).

This imposes an upper limit on the Bag Constant, $B_{max}$, for a given $\Delta$ and $m_s$. The limit is calculated by solving the EoS at the chemical potential threshold $\mu_{limit} = M_n / 3$:

$$
B_{max} \approx \frac{3}{4\pi^2}\mu_{limit}^4 + \frac{3}{\pi^2}\left(\Delta^2 - \frac{m_s^2}{4}\right)\mu_{limit}^2
$$

If a sampled $B$ exceeds $B_{max}$, the matter would be unstable relative to neutrons; such configurations are discarded.

### 3. Inverse Scaling Law
To efficiently explore the parameter space, an inverse scaling relationship is utilized. The maximum mass of a quark star scales inversely with the square root of the Bag Constant:

$$
M_{max} \propto \frac{1}{\sqrt{B}} \quad \Longrightarrow \quad B \propto \frac{1}{M_{max}^2}
$$

This relation allows the algorithm to target a specific maximum mass by solving for the required $B$. A correction factor is applied to account for the stiffening effect of the pairing gap $\Delta$:

$$
B_{target} \approx B_{ref} \left( \frac{M_{ref}}{M_{target}} \right)^2 \times \left( 1 + \gamma \frac{\Delta}{\Delta_{ref}} \right)
$$

## Algorithm and Calculations

1.  **Microphysics Sampling**
    The pairing gap $\Delta$ and strange quark mass $m_s$ are sampled uniformly from physically motivated ranges (e.g., $\Delta \in [57, 250]$ MeV).

2.  **Stability Constraint**
    The maximum allowable Bag Constant ($B_{max}$) is calculated based on the stability criterion relative to the neutron mass. If the parameter space is invalid (i.e., $B_{max} < B_{min}$), the sample is rejected.

3.  **Target-Driven $B$ Selection**
    Instead of sampling $B$ uniformly (which would heavily bias the dataset toward low-mass stars), a **target maximum mass** is selected from a uniform distribution (e.g., $1.0$ to $4.0 M_{\odot}$). The inverse scaling law is used to compute the $B$ required to achieve this mass. Random noise is added to the result to explore the local parameter space.

4.  **Structure Integration**
    The resulting parameters ($B, \Delta, m_s$) are passed to the TOV solver. The star's structure (Mass, Radius, Tidal Deformability) is integrated from the center to the surface.

5.  **Validation and Feature Extraction**
    *   **Mass Filter:** Stars with maximum masses outside the target range are discarded.
    *   **Radius Filter:** Extremely diffuse solutions ($R > 22$ km) are rejected.
    *   **Interpolation:** The discrete Mass-Radius sequence is interpolated to extract specific features at canonical masses (e.g., $1.4 M_{\odot}$).
    *   **Slope Calculation:** The topological slope of the branch, $dR/dM$, is computed via finite differences at intervals of $0.2 M_{\odot}$.

## Inputs and Outputs

### Inputs
*   **Constants**: Physical ranges for $\Delta$, $m_s$, and $B$ (defined in `const.py`).
*   **Control Parameters**: Batch size and random seed.

### Outputs
*   **Synthetic Dataset**: A list of data vectors representing stable Quark Stars. Each vector contains:
    *   **Macroscopic:** Mass, Radius, Tidal Deformability ($\Lambda$).
    *   **Microscopic:** $B$, $\Delta$, $m_s$, Central Density, Sound Speed.
    *   **Topology:** Slopes at $1.4, 1.6, 1.8, 2.0 M_{\odot}$.
    *   **Label:** 1 (Quark).

## Connection to the Overall Workflow
This script operates in parallel with the hadronic generation worker. While the hadronic worker explores the discrete space of nuclear models, this worker explores the continuous parameter space of QCD phase diagrams. The combined output forms the balanced dataset used to train the machine learning classifiers to distinguish between the two states of matter.
