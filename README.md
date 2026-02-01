# ML_Classification_NS
ML pipeline distinguishing Hadronic vs. Quark Stars. Generates 20k+ synthetic EoS (CFL/Nuclear), solves TOV equations, and trains Random Forests to break geometric degeneracy using Tidal Deformability. Applied to GW170817 &amp; HESS J1731.




# Machine Learning Classification of Neutron Star Composition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Thesis-orange)

A computational framework to distinguish **Hadronic Stars** (gravity-bound) from **Strange Quark Stars** (self-bound) using supervised machine learning. This project resolves the "Masquerade Problem"—where distinct phases of matter produce degenerate mass-radius relations—by integrating **Tidal Deformability ($\Lambda$)** and topological stability features.

##  Project Overview

The internal composition of neutron stars remains one of the greatest unknowns in astrophysics. This pipeline implements a **Forward Modeling** approach:
1.  **Generate** a massive synthetic library of ~20,000 physical Equation of State (EoS) families.
2.  **Solve** the Tolman-Oppenheimer-Volkoff (TOV) and Riccati equations to map microphysics to macroscopic observables ($M, R, \Lambda$).
3.  **Train** a hierarchical Random Forest classifier to learn the decision boundary between Hadronic and Quark matter.
4.  **Infer** the composition of real astrophysical candidates (GW170817, HESS J1731-347) using Monte Carlo error propagation.

##  Key Features

### 1. Physics Engine (`src/physics`)
*   **Hadronic Models:** Uses spectral mixing of 21 baseline nuclear theories (APR, MDI, SLy, RMF) with homologous scaling to cover the parameter space.
*   **Quark Models:** Implements the **Generalized Color-Flavor-Locked (CFL)** parametrization ($B, \Delta, m_s$) with a dynamic QCD stability window.
*   **Relativistic Solver:** Custom RK45 integrator for TOV structure and Tidal Deformability, enforcing **Causality** ($c_s^2 \le 1$) and **Stability** ($dM/dP_c > 0$).

### 2. Machine Learning Pipeline (`src/ml_pipeline`)
*   **Data Strategy:** Generates ~750,000 stellar configurations. Uses **Grouped Cross-Validation** (by `Curve_ID`) to prevent data leakage from "sibling" stars.
*   **Model Hierarchy:** Trains distinct models to quantify information gain:
    *   `Model Geo`: Mass + Radius (Baseline).
    *   `Model A`: Mass + Radius + Tidal Deformability (GW standard).
    *   `Model D`: Full topology (includes slope $dR/dM$).
*   **Calibration:** Isotonic regression ensures output probabilities are statistically reliable.

### 3. Visualization Suite (`src/visualize`)
*   Produces publication-quality plots (PDF/HTML).
*   **Grand Summary Triptych:** EoS bands, M-R Manifolds, and Tidal trends.
*   **Diagnostics:** ROC curves, Confusion Maps, and Partial Dependence Plots.
*   **3D Manifolds:** Interactive visualization of the $M-R-\Lambda$ phase space.

##  Repository Structure

```bash
.
├── main.py                   # Orchestration script (Data Gen -> Train -> Plot)
├── data/                     # Generated synthetic datasets (CSV)
├── plots/                    # Output figures (PDFs)
├── src/
│   ├── const.py              # Physical constants & constraints
│   ├── physics/              # Physics kernel (EoS, TOV, CFL)
│   ├── ml_pipeline/          # RF Training, Validation, Inference
│   └── visualize/            # Plotting logic & Style config
└── Thesis.tex                # (Optional) Source for the manuscript
```

##  Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/neutron-star-ml.git
    cd neutron-star-ml
    ```

2.  Install dependencies:
    ```bash
    pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm
    ```

##  Usage

Run the full pipeline (Generation, Training, and Visualization) with a single command:

```bash
python main.py
```

*Note: The first run will trigger the parallel generation of 20,000 EoS curves, which may take 1-2 hours depending on CPU cores. Subsequent runs will load the cached `data/thesis_dataset.csv`.*

## 📊 Key Results

### Classification Accuracy
The analysis confirms that **Tidal Deformability** is the critical discriminator required to break the geometric degeneracy.

| Model | Features | Accuracy | AUC |
| :--- | :--- | :--- | :--- |
| **Geo** | Mass, Radius | 86.8% | 0.941 |
| **Model A** | Mass, Radius, $\log\Lambda$ | **97.3%** | **0.996** |
| **Model D** | + Slope $dR/dM$ | 100.0% | 1.000 |

### Astrophysical Inference
Applying the model to real candidates via Monte Carlo sampling ($N=5000$):

*   **GW170817:** $P(\text{Quark}) \approx 41.5\%$ (Inconclusive / Likely Hadronic).
*   **HESS J1731-347:** $P(\text{Quark}) \approx 68.3\%$ (Likely **Quark** due to the "Lightness Problem").
*   **PSR J0740+6620:** $P(\text{Quark}) \approx 33.1\%$ (Likely Hadronic).

## 👥 Credits & Acknowledgements

**Author:** Ioannis Papathanasiou 
**Supervisors:** 
*   **Prof. Charalampos Moustakidis** 
*   **Theodoros Diakonidis** 

This work was conducted at the **Aristotle University of Thessaloniki**, Department of Physics.

---
*For more details, please refer to the full thesis PDF or the `src/` documentation.*
