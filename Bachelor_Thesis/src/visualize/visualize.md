# Visualization Suite Documentation

## Overview
This document details the visualization modules used in the thesis project. These scripts transform raw simulation data and machine learning predictions into figures. The visualization suite is divided into three main categories:
1.  **Physics Verification:** validating the generated Equations of State (EoS) against theoretical constraints.
2.  **Manifold Visualization:** mapping the distributions of stars in macroscopic and microscopic phase spaces.
3.  **Machine Learning Diagnostics:** interpreting the performance and decision-making processes of the classifiers.

---

# 1. Configuration and Style

## style_config.py

### Role in the Project
This script establishes the global graphical standards for the entire project. 

### Design Standards
Instead of default settings, specific parameters are enforced to maximize readability and compatibility:
*   **Typography:** Computer Modern fonts (Type 1/TrueType) are used for all text and mathematical symbols ($...$) to match standard LaTeX documents.
*   **Color Palette:** A custom dictionary defines high-contrast colors for distinguishing populations:
    *   **Hadronic (Standard):** Green tones.
    *   **Quark (CFL):** Magenta tones.
    *   **Constraints:** Black/Gray for observational limits (e.g., causality, unitarity).
    *   **Errors:** Gold/Blue for False Negatives/Positives.
*   **Geometry:** Figures are sized for standard page widths, with inward-facing ticks and high resolution (300 DPI).

---

# 2. Physics Verification Plots

## plot_theoretical_eos.py

### Role in the Project
This script visualizes the "Priors" of the simulation. It generates a spaghetti plot of Pressure versus Energy Density ( $P(\epsilon)$ ) for a random subset of the generated models. This confirms that the synthetic dataset covers the physically relevant phase space and respects fundamental bounds.

### Physics and Equations
The Equation of State relates pressure$P$to energy density$\epsilon$. Two fundamental physical limits are plotted for reference:
1.  **Causality Limit:** The stiffest possible matter allowed by special relativity, where the speed of sound equals the speed of light ($c_s = 1$).
   $P = \epsilon$
2.  **Conformal Limit:** The theoretical asymptotic limit for non-interacting quarks at infinite density.
   $P = \frac{1}{3}\epsilon \quad \implies \quad c_s^2 = \frac{1}{3}$

### Algorithm
*   A set of random seeds is selected.
*   The `worker_get_plot_curve` module is invoked in parallel to generate high-resolution$P-\epsilon$grids for both Hadronic and Quark models.
*   The curves are plotted on a log-log scale.
*   The theoretical limits are overlaid to verify that no generated model violates causality (crosses the $P=\epsilon$ line).

## plot_stability_window.py

### Role in the Project
This script verifies that the generated Quark Star models reside within the theoretical "Stability Window" of Quantum Chromodynamics (QCD). It plots the Vacuum Pressure ($B$) against the Pairing Gap ($\Delta$) and Strange Quark Mass ($m_s$).

### Physics and Equations
For Strange Quark Matter to be the true ground state of strong interactions (the Bodmer-Witten hypothesis), it must be:
1.  **Stable relative to Iron:** The energy per baryon at zero pressure must be less than $930$ MeV. This sets a **lower bound** on the Bag Constant$B$.
2.  **Unstable relative to Neutrons (at low density):** Ordinary nuclei must not spontaneously decay into quark matter. This sets an **upper bound** on$B$, which depends on the gap $\Delta$:
   $B_{max}(\Delta) \approx \frac{3}{4\pi^2}\mu_n^4 + \frac{3}{\pi^2}\Delta^2 \mu_n^2$
    where$\mu_n$is the neutron chemical potential.

### Algorithm
*   The unique Quark models are filtered from the dataset.
*   They are scattered on a$B$vs.$\Delta$plane.
*   Analytic stability boundaries are calculated and overlaid as shaded regions ("Forbidden Zones").
*   The plot confirms that all generated points fall within the allowed triangle of stability.

## plot_surface_density.py

### Role in the Project
This script demonstrates the fundamental difference in surface boundary conditions between the two classes of stars. It highlights the "Forbidden Gap" in density space that separates gravity-bound stars from self-bound stars.

### Physics and Equations
*   **Hadronic Stars:** Have a crust made of iron nuclei. The density at the surface drops to zero (or negligible atomic density) as pressure vanishes.
   $\epsilon_{surf} \approx 0$
*   **Quark Stars:** Are "self-bound" by the strong interaction. At the surface ( $P=0$ ), the density remains finite and high, approximately four times the Bag Constant.
   $\epsilon_{surf} \approx 4B$

### Algorithm
*   The surface density ($\epsilon_{surf}$) is extracted for all stars.
*   A Kernel Density Estimation (KDE) is plotted for the Quark population, showing a peak around$400-600$MeV/fm$^3$.
*   The Hadronic population is represented as a vertical line at $\epsilon=0$.
*   The region between$0$and the minimum quark density is shaded as the "Forbidden Region."

---

# 3. Manifold Visualization (The "Grand Results")

## plot_grand_summary.py

### Role in the Project
This is the primary result figure for the thesis. It aggregates the entire dataset into three panels, visualizing the statistical properties of the simulated population with $5^{th}-95^{th}$ percentile confidence bands.

### Physics and Equations
The plot consists of three panels corresponding to the key equations of stellar structure:
1.  **Equation of State:** $P$vs.$\epsilon$ (Log-Log). Shows the stiffness of matter.
2.  **Mass-Radius Relation:** $M$vs.$R$ . Defined by the TOV equilibrium.
   $\frac{dP}{dr} = -\frac{G\epsilon m}{r^2} \dots$
3.  **Tidal Deformability:** $\Lambda$ vs. $M$ . Defined by the Riccati equation for metric perturbation.

### Algorithm
*   **Interpolation:** Because every star has a different central pressure, the curves are interpolated onto a common grid of densities and masses.
*   **Statistics:** At each grid point, the median, $5^{th}$ percentile, and $95^{th}$ percentile are calculated.
*   **Plotting:** These statistics are rendered as shaded confidence bands (solid for Hadronic, hatched for Quark).
*   **Constraints:** Observational limits (GW170817, PSR J0740) are overlaid to show consistency with reality.

## plot_statistical_bands.py

### Role in the Project
This script is a variant of the Grand Summary that focuses specifically on producing clean, smoothed statistical bands. It is used to generate the "Theoretical Prior" figure, emphasizing the overlap and distinct regions of the two populations.

### Algorithm
*   Similar to `plot_grand_summary`, it aggregates curves by ID.
*   It applies a Gaussian smoothing filter to the calculated percentiles to remove numerical noise from the interpolation steps, producing aesthetically clean contours for the final document.

## plot_physics_manifold.py

### Role in the Project
This script visualizes the "Phase Space" of neutron stars using 2D probability density contours. It is presented as a triptych (three panels):
1.  Hadronic Population Only.
2.  Quark Population Only.
3.  Intersection/Overlay.

### Algorithm
*   **Kernel Density Estimation (KDE):** The density of stars in the Mass-Radius plane is estimated using a Gaussian kernel.
*   **Contouring:** Isodensity contours (e.g., enclosing 99%, 50%, and 10% of the population) are drawn.
*   **Overlay:** The intersection panel highlights the "Confusion Region" where the two populations overlap, indicating where classification is physically difficult or impossible based on Mass and Radius alone.

## plot_3d_separation.py

### Role in the Project
This script extends the manifold visualization into 3D space, plotting Mass ( $M$ ), Radius ( $R$ ), and Tidal Deformability ($\Lambda$) simultaneously. It proves that while populations may overlap in 2D projections (e.g.,$M-R$), they are often separable in the full 3D higher-dimensional space.

### Algorithm
*   A 3D scatter plot is generated.
*   **Marginal Projections:** The 3D cloud is projected onto the "walls" of the plot (xy, xz, yz planes) using KDE. This creates "shadows" that represent the 2D correlations (e.g., Mass-Radius, Radius-Tidal).
*   Interactive HTML versions (via Plotly) are optionally generated for data exploration.

## plot_microphysics_3d.py

### Role in the Project
This script visualizes the internal **microphysical** phase space, rather than the observable macroscopic space. The axes are:
1.  **Central Density ($\epsilon_c$):** The core compression.
2.  **Sound Speed ($c_s^2$):** The stiffness.
3.  **Slope ($dR/dM$):** The topological stability.

### Algorithm
*   Similar to the macroscopic 3D plot, stars are scattered in this parameter space.
*   Projections are drawn on the walls to visualize correlations (e.g., how the speed of sound correlates with the slope of the stability curve). This confirms that "stable" quark stars occupy a distinct topological region compared to hadronic stars.

---

# 4. Topological Diagnostics

## plot_slope_evolution.py

### Role in the Project
This script investigates the "Topological Phase Transition." It plots the local slope of the Mass-Radius curve ($dR/dM$) against the central speed of sound ($c_s^2$).

### Physics and Equations
*   **Hadronic Branch:** Typically exhibits a negative slope ( $dR/dM < 0$ ). As mass increases, the star compresses and the radius shrinks.
*   **Quark Branch:** Often exhibits a positive slope ( $dR/dM > 0$ ) due to the stabilizing effect of the vacuum pressure$B$.
*   **Stability Criterion:** A slope of zero ( $dR/dM = 0$ ) marks the transition to instability (maximum mass).

### Algorithm
*   The dataset is filtered to extract slope and sound speed values at specific canonical masses ($1.4, 1.6, 1.8, 2.0 M_{\odot}$).
*   A scatter plot is generated.
*   Vertical lines mark the conformal limit ( $c_s^2 = 1/3$ )  and causal limit ( $c_s^2 = 1$ ).
*   The plot reveals distinct clustering, showing that Quark stars preserve stability (positive slope) even at high sound speeds.

## plot_slope_vs_radius.py

### Role in the Project
This script visualizes the slope$dR/dM$explicitly against the Radius. It demonstrates that for a fixed radius, the two types of matter respond differently to the addition of mass.

### Algorithm
*   Data is extracted at fixed mass steps (e.g.,$1.4 M_{\odot}$).
*   Slope is plotted on the Y-axis, Radius on the X-axis.
*   The "Zero Slope" line is drawn to separate the standard branch from the stable branch.

---

# 5. Machine Learning Diagnostics

## plot_diagnostics.py

### Role in the Project
This script evaluates the standard performance metrics of the trained Machine Learning classifiers.

### Figures Produced
1.  **ROC Curves:** Receiver Operating Characteristic curves are plotted for the hierarchy of models (Geo, A, B, C, D). The Area Under the Curve (AUC) demonstrates the improvement in discrimination power as physical features are added.
2.  **Calibration Curve (Reliability Diagram):** Plots Predicted Probability vs. True Fraction of Positives. This validates that a predicted probability of 80% genuinely corresponds to an 80% success rate.
3.  **Violin Plots:** Shows the distribution of radii for Hadronic vs. Quark stars in different mass bins, illustrating why classification is easier at high masses (distributions separate) and harder at low masses (distributions overlap).

## plot_advanced_diagnostics.py

### Role in the Project
This script performs deeper analysis into specific failure modes and physical correlations.

### Sub-Analyses
1.  **Misclassification Map:**
    *   The test set is plotted in the Mass-Radius plane.
    *   Incorrect predictions are highlighted (False Positives and False Negatives).
    *   This reveals the "Geography of Failure"—showing that errors are concentrated at the interface where the two populations overlap.
2.  **Universal Relations (I-Love-Q):**
    *   Plots Compactness ( $C = GM/Rc^2$ )  against Tidal Deformability ( $\Lambda$ ).
    *   Checks if the generated stars follow the expected universal tracks for compact objects.

## plot_pdp.py

### Role in the Project
This script generates Partial Dependence Plots (PDP) to interpret the "Black Box" of the Random Forest.

### Algorithm
*   The marginal effect of a single feature (e.g., Radius) on the predicted probability$P(\text{Quark})$is computed by integrating out all other features.
*   **Model A (Observables):** Shows how probability changes with Mass and Radius.
*   **Model D (Physics):** Shows how probability changes with Sound Speed and Slope. This confirms that the model has learned the correct physical associations (e.g., high sound speed increases the probability of being a Quark star).

## plot_physical_insights.py

### Role in the Project
This script ranks the importance of different physical features and visualizes the connection between the Equation of State trajectories and the resulting stellar cores.

### Figures Produced
1.  **Feature Importance:** A bar chart comparing the "Gini Importance" of macroscopic observables (Mass, Radius) versus microscopic parameters (Slope, Sound Speed). It quantitatively shows that topological features are highly predictive.
2.  **Speed of Sound Trajectories:** Plots the $c_s^2$vs$\epsilon$ curves and overlays the actual central conditions of the stars. This visualizes where the population lives relative to the conformal limit ( $c_s^2=1/3$ ).

## plot_corner.py

### Role in the Project
This script generates pairwise scatter plots (Corner plots) for the dataset features.

### Figures Produced
1.  **Macro Corner:** Radius vs. Mass vs. Tidal Deformability. Visualizes the observational correlations.
2.  **Micro Corner:** Density vs. Sound Speed vs. Slope. Visualizes the internal physical correlations.
*   Diagonal panels show the marginal distribution (1D histogram/KDE) for each feature.
*   Off-diagonal panels show the 2D correlations, highlighting how the two classes separate in high-dimensional space.

## plot_correlations.py

### Role in the Project
This script explicitly links the microscopic world to the macroscopic world. It generates scatter plots showing how an internal parameter determines an observable property.

### Figures Produced
1.  **Density vs. Radius:** Shows how central compression dictates stellar size.
2.  **Stiffness vs. Max Mass:** Shows how the speed of sound ($c_s^2$) correlates with the maximum mass capacity of the star.
3.  **Slope vs. Tidal Deformability:** Links the topological stability derivative to the tidal response $\Lambda$.

---

## Connection to Workflow
All visualization scripts depend on the `thesis_dataset.csv` generated by the physics engine and the trained models produced by the machine learning pipeline. They are triggered automatically by the `main.py` orchestrator at the end of a successful run, depositing all figures into the `plots/` directory.
