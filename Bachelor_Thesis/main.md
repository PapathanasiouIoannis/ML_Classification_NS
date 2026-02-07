# Thesis Pipeline Orchestrator

## Role in the Project
This script serves as the central executive interface for the entire research project. It orchestrates the end-to-end workflow, connecting the fundamental physics simulations (Equation of State generation and TOV integration) with the data science pipeline (machine learning training and statistical analysis).

Its primary function is to manage the lifecycle of the synthetic dataset: it determines whether new physical simulations are required, distributes the computational load across processor cores, enforces statistical rigor through class balancing, and triggers the generation of all diagnostic figures and final thesis plots.

## Physics and Equations
While the specific differential equations for stellar structure are solved in the subordinate worker modules, this script defines the **statistical ensemble** used for inference.

### Statistical Prior Construction
The pipeline is designed to produce a dataset that approximates a uniform prior over the model space. The dataset $\mathcal{D}$ consists of $N$ observations, where each observation represents a stable neutron star configuration:

$$
\mathcal{D} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^{N}
$$

Here, $\mathbf{x}_i$ represents the vector of macroscopic observables and microscopic parameters (e.g., Mass $M$, Radius $R$, Tidal Deformability $\Lambda$, Central Density $\epsilon_c$), and $y_i$ is the binary label:

$$
y_i = 
\begin{cases} 
0 & \text{if Hadronic (Standard Matter)} \\
1 & \text{if Quark (CFL Color Superconductivity)}
\end{cases}
$$

### Class Balancing
To prevent bias in the machine learning classifiers, the dataset is forced into a balanced distribution. If the generation process yields $N_H$ hadronic stars and $N_Q$ quark stars, the final dataset size $N_{final}$ is determined by the limiting class:

$$
N_{final} = 2 \times \min(N_H, N_Q)
$$

The majority class is randomly undersampled to match the minority class, ensuring that the baseline probability of any random star being a quark star is exactly $0.5$ ($P(Q) = P(H) = 50\%$). This removes the "class imbalance" prior, forcing the machine learning models to learn physical features rather than statistical frequencies.

## Algorithm and Calculations

The workflow proceeds in a strictly sequential manner:

1.  **Environment Initialization**
    Directory structures for data storage and graphical outputs are established.

2.  **Physics Calibration (Baseline Calculation)**
    Before generating mixed hadronic models, the maximum stable mass $M_{max}$ is calculated for every pure parent Equation of State in the library. These baseline values are required to normalize the homologous scaling factors used later in the generation process.

3.  **Parallel Data Generation**
    If a pre-computed dataset is not found, the simulation engine is engaged.
    *   A target number of Equation of State (EoS) curves is defined.
    *   The workload is split into batches.
    *   Computational tasks are distributed in parallel across all available CPU cores. One set of workers generates Hadronic stars (using parametric mixing), while another generates Quark stars (using the Generalized CFL model).
    *   The results are aggregated into a single master table.

4.  **Data Processing**
    The raw simulation data is cleaned and balanced. The resulting dataset is shuffled to remove any ordering artifacts from the generation phase.

5.  **Machine Learning Training**
    The balanced dataset is passed to the training module. A hierarchy of Random Forest classifiers is trained, ranging from simple geometric models (Mass-Radius only) to complex physics-informed models (including topology and sound speed).

6.  **Visualization and Analysis**
    Once models are trained, a comprehensive suite of plotting routines is executed. This includes:
    *   **Diagnostic Plots:** Confusion matrices, ROC curves, and reliability diagrams.
    *   **Physics Manifolds:** Visualizations of the Mass-Radius and Tidal Deformability relations.
    *   **Microphysics Checks:** Verification of the QCD stability window and speed of sound behavior.
    *   **Inference:** Application of the trained models to real astrophysical candidates (e.g., GW170817, PSR J0740+66).

## Inputs and Outputs

### Inputs
*   **Physics Constants**: Fundamental constants (nuclear saturation density, conversion factors) are loaded from the configuration.
*   **EoS Library**: A set of analytic parameterizations for nuclear matter.

### Outputs
*   **Dataset File**: A CSV file containing thousands of simulated stars, labeled by their physical composition.
*   **Trained Models**: A dictionary of optimized Random Forest classifiers ready for inference.
*   **Figure Set**: A collection of PDF plots characterizing the physics and the performance of the machine learning models.

## Connection to the Overall Workflow
This script is the **driver** of the project. It does not perform the numerical integration itself but calls the functions that do. It is responsible for creating the `thesis_dataset.csv` file, which is the foundational data source for every subsequent analysis step, including model training, hyperparameter tuning, and the final physical interpretation of results.
