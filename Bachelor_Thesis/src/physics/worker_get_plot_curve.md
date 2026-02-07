# Single Curve Generator for Visualization

## Role in the Project
This script is a specialized worker module designed to generate individual, high-resolution Equation of State (EoS) curves for visualization purposes. Unlike the main data generation workers (which focus on bulk statistics and scalar features like $R_{1.4}$), this worker returns the full functional form of the EoS—specifically the Pressure vs. Energy Density relation ($P(\epsilon)$)—and the complete Mass-Radius sequence.

It is primarily used by the plotting routines (e.g., `plot_theoretical_eos.py`) to render the "priors" of the model space, visualizing how the generated Hadronic and Quark populations cover the physical phase space compared to theoretical constraints (causality, conformal limit).

## Physics and Equations

### 1. Hadronic Generation Logic
The logic for generating a single hadronic curve mirrors the bulk generation process but focuses on extracting the dense EoS grid.

*   **Model Mixing:** Two parent models (e.g., SLy, APR) are selected and mixed with a weight $w$.
*   **Homologous Scaling:** A scaling factor $\alpha$ is calculated to shift the maximum mass of the sequence to a target value $M_{target}$.
    $$
    P_{new} = \alpha P_{old}, \quad \epsilon_{new} = \alpha \epsilon_{mix}(P_{old})
    $$
*   **Crust-Core Transition:** The transition from the low-density crust to the high-density core is handled via a randomized transition pressure $P_{trans}$, ensuring the crust physics (SLy) matches the core smoothly.

### 2. Quark Generation Logic (CFL)
The logic for generating a quark curve follows the Target-Driven Inverse Sampling method.

*   **Parameter Selection:** The pairing gap $\Delta$ and strange quark mass $m_s$ are sampled from their physical priors.
*   **Stability Check:** The maximum allowed Bag Constant $B_{max}$ is computed to ensure stability relative to neutrons.
*   **Inverse Sampling:** A target maximum mass is chosen, and the required Bag Constant $B$ is inferred using the inverse scaling law:
    $$
    B \propto \frac{1}{M_{target}^2}
    $$
*   **Algebraic EoS:** The pressure $P$ and energy density $\epsilon$ are related algebraically through the quark chemical potential $\mu$:
    $$
    P(\mu) = \frac{3}{4\pi^2}\mu^4 + \frac{3}{\pi^2}\left(\Delta^2 - \frac{m_s^2}{4}\right)\mu^2 - B
    $$

## Algorithm and Calculations

1.  **Initialization**
    The worker accepts a `mode` ('hadronic' or 'quark') and a random `seed`.

2.  **Trial-and-Error Loop**
    A `while True` loop is employed to ensure that only physically valid curves are returned.
    *   **Parameter Sampling:** Random parameters ($\alpha, w$ or $B, \Delta, m_s$) are drawn.
    *   **TOV Integration:** The stellar structure equations are solved.
    *   **Validation:**
        *   The maximum mass must fall within the target range (e.g., $2.0 - 3.0 M_{\odot}$).
        *   The radius must be within observational bounds ($R < 14$ km for hadronic, $R < 22$ km for quark).
        *   **Causality Check:** The speed of sound must not exceed the speed of light ($c_s \le 1$) at relevant densities.

3.  **Dense Grid Generation**
    Once a valid parameter set is found, the script generates a high-resolution grid for plotting:
    *   A logarithmic pressure grid $P_{grid}$ is defined (spanning $10^{-4}$ to $3000$ MeV/fm$^3$).
    *   For **Hadronic models**, the mixing and scaling laws are applied to compute $\epsilon$ at each $P$ point.
    *   For **Quark models**, the algebraic CFL relation is inverted (solving for $\mu$) to find $\epsilon$ at each $P$ point.

## Inputs and Outputs

### Inputs
*   **mode**: A string ('hadronic' or 'quark') determining the physics engine.
*   **baselines**: A dictionary of parent model maximum masses (used only for hadronic scaling).
*   **seed**: An integer for random number generation.

### Outputs
*   **Tuple `(curve, eos_arr)`**:
    *   `curve`: The Mass-Radius sequence (a list of $[M, R, \Lambda, \dots]$ points).
    *   `eos_arr`: A 2D array where Column 0 is Energy Density $\epsilon$ and Column 1 is Pressure $P$. This represents the "equation of state" curve itself.

## Connection to the Overall Workflow
This script is a **visualization helper**. It is not used to generate the training data for the machine learning model. Instead, it is called by `plot_theoretical_eos.py` to produce the background "spaghetti plots" that illustrate the theoretical uncertainty bands of the equation of state. It allows the thesis figures to accurately reflect the exact physics used in the training set.
