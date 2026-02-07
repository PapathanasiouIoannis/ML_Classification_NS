# Analytic Equation of State Library (Hadronic & Crust)

## Role in the Project
This script serves as the foundational database of nuclear physics for the hadronic sector of the project. It provides a collection of analytic parameterizations that describe the Equation of State (EoS) for dense nuclear matter.

Specifically, it defines the functional relationship between pressure and energy density, $\epsilon(P)$, for a variety of theoretical models (e.g., SLy, APR, MDI). Furthermore, it analytically computes the derivatives required to determine the speed of sound within the star. These functions are consumed by the hydrostatic equilibrium solvers to generate mass-radius sequences for hadronic neutron stars.

## Physics and Equations

### The Equation of State
The Equation of State relates the microscopic pressure $P$ to the energy density $\epsilon$. In this library, the EoS is parameterized analytically to fit tabulated data from nuclear many-body theory (typically based on the spectral fits by Read et al., 2009).

For the high-density core, the energy density is modeled as a function of pressure:

$$
\epsilon(P) = \sum_{i} c_i P^{\gamma_i} + \sum_{j} d_j \left( 1 - e^{-P/\lambda_j} \right)
$$

Here:
*   $P$ is the pressure in $\text{MeV}/\text{fm}^3$.
*   $\epsilon$ is the energy density in $\text{MeV}/\text{fm}^3$.
*   $c_i, d_j, \lambda_j$ are fitting coefficients specific to each nuclear model (e.g., "SLy", "APR-1").
*   $\gamma_i$ are adiabatic indices governing the stiffness of the matter.

### The Speed of Sound
To solve the Tolman–Oppenheimer–Volkoff (TOV) equations, the sound speed profile of the star is required. The squared speed of sound, $c_s^2$, is defined by the thermodynamic derivative:

$$
c_s^2 = \frac{dP}{d\epsilon} = \left( \frac{d\epsilon}{dP} \right)^{-1}
$$

By defining $\epsilon(P)$ analytically using symbolic mathematics, the derivative $d\epsilon/dP$ is computed exactly, avoiding numerical noise associated with finite-difference methods. This ensures smooth and stable integration of the stellar structure.

### Crust Physics (SLy Model)
The outer layers of the neutron star (the crust) are modeled using the Douchin & Haensel (SLy) EoS. The crust is divided into four distinct regions based on density, representing the transition from the outer envelope to the inner crust.

The outermost envelope (Region 4) is modeled using a logarithmic polynomial fit:

$$
\log_{10}(\epsilon) = \sum_{k=0}^{5} a_k (\log_{10} P)^k
$$

This captures the complex behavior of the atomic lattice and electron gas in the low-density regime.

## Algorithm and Calculations

1.  **Symbolic Definition:**
    A set of analytic expressions is defined using symbolic variables. Each expression corresponds to a specific nuclear physics model (e.g., "MDI-1", "H4", "MS1"). These models cover a wide range of stiffnesses, representing the theoretical uncertainty in the behavior of supranuclear matter.

2.  **Analytic Differentiation:**
    For every EoS model, the derivative of energy density with respect to pressure is computed symbolically. This provides the inverse of the squared sound speed, which is a critical term in the relativistic hydrostatic equations.

3.  **Numerical Compilation:**
    The symbolic expressions for both the function $\epsilon(P)$ and its derivative $\epsilon'(P)$ are compiled into optimized numerical functions. This allows the TOV solver to evaluate the EoS and sound speed efficiently millions of times during the generation of the synthetic dataset.

4.  **Library Aggregation:**
    The resulting functions are organized into dictionaries:
    *   **Core Functions:** Containing the high-density behaviors for various hadronic models.
    *   **Crust Functions:** Containing the four-layer model for the low-density crust.

## Inputs and Outputs

### Inputs
*   This script does not require external data inputs. It relies on hardcoded parameter sets derived from established nuclear physics literature (e.g., Read et al. 2009).

### Outputs
*   **Core Library:** A dictionary mapping model names (e.g., "SLy", "APR") to tuples of callable functions `(epsilon_func, derivative_func)`.
*   **Crust Library:** A dictionary mapping crust layer identifiers (e.g., "c1", "c2") to their corresponding EoS functions.

## Connection to the Overall Workflow
This library provides the **physics primitives** used by the `worker_hadronic_gen.py` script. When a new hadronic star is simulated:
1.  Two "parent" models are selected from this library.
2.  Their pressure-energy relations and derivatives are retrieved.
3.  These functions are mixed and scaled to create a unique, randomized EoS, which is then solved to produce the Mass-Radius relation.
