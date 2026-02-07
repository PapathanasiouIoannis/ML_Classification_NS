# Machine Learning Model Training (Random Forest Hierarchy)

## Role in the Project
This script executes the central classification task of the thesis: distinguishing between Hadronic stars (Standard Model matter) and Quark stars (Color-Flavor-Locked matter) based on their physical properties.

A hierarchy of machine learning models is trained, ranging from a purely observational baseline (using only Mass and Radius) to a fully physics-informed model (incorporating internal thermodynamics and topology). This hierarchical approach allows the physical information content of specific features—such as the speed of sound or the slope of the mass–radius relation—to be quantified by observing the improvement in classification accuracy as new features are added.

## Physics and Equations

### 1. The Classification Problem
The problem is formulated as a binary classification task. For a given neutron star configuration characterized by a feature vector $\mathbf{x}$, the goal is to estimate the probability $P(Q|\mathbf{x})$ that the star contains quark matter.

The class labels $y$ are defined as:
$$
y = 
\begin{cases} 
0 & \text{Hadronic (Standard)} \\
1 & \text{Quark (CFL)}
\end{cases}
$$

### 2. Feature Hierarchy
Five distinct models are trained, each with access to a different subset of physical information. This simulates different levels of observational and theoretical knowledge.

*   **Model Geo (Geometric):**
    $$ \mathbf{x}_{\text{Geo}} = [M, R] $$
    Uses only Mass ($M$) and Radius ($R$). This represents the capabilities of X-ray pulse profiling (e.g., NICER).

*   **Model A (Observational):**
    $$ \mathbf{x}_{\text{A}} = [M, R, \log_{10}(\Lambda)] $$
    Adds Tidal Deformability ($\Lambda$). This represents the capabilities of Gravitational Wave detectors (LIGO/Virgo).

*   **Model B (Density):**
    $$ \mathbf{x}_{\text{B}} = [M, R, \log_{10}(\Lambda), \epsilon_c] $$
    Adds Central Energy Density ($\epsilon_c$). This requires knowledge of the central boundary condition.

*   **Model C (Stiffness):**
    $$ \mathbf{x}_{\text{C}} = [\dots, c_s^2] $$
    Adds the Central Speed of Sound squared ($c_s^2$). This probes the stiffness of the Equation of State.

*   **Model D (Topological / Full Physics):**
    $$ \mathbf{x}_{\text{D}} = [\dots, dR/dM|_{1.4}] $$
    Adds the slope of the Mass-Radius curve ($dR/dM$). This captures the "Topological Phase Transition," distinguishing the generic unstable branch of hadronic stars from the stable branch of quark stars.

### 3. Probability Calibration
Standard Random Forests output a "score" (the fraction of trees voting for a class), which is not necessarily a well-calibrated probability. To ensure that a predicted probability of $0.8$ implies an $80\%$ chance of being a Quark star, **Isotonic Regression** is applied.

The uncalibrated score $s$ is mapped to a probability $p$ via a non-decreasing function $f$:
$$
p = f(s)
$$
This ensures that the output $P(Q|\mathbf{x})$ can be interpreted as a rigorous physical confidence level.

## Algorithm and Calculations

1.  **Data Preparation**
    The dataset is loaded and cleaned. The logarithmic tidal deformability is computed:
    $$ \log_{10}(\Lambda) = \ln(\Lambda) / \ln(10) $$
    Rows with missing physical data (e.g., stars that collapsed before reaching the reference mass for slope calculation) are removed.

2.  **Group-Aware Splitting**
    A standard random split is **physically invalid** here because points generated from the same Equation of State (EoS) curve are highly correlated. To prevent data leakage (where the model memorizes the EoS rather than learning the physics), a **Group Shuffle Split** is employed.
    *   The "Groups" are defined by the unique `Curve_ID`.
    *   If an EoS curve is selected for the Training Set, *all* stars belonging to that EoS are placed in Training.
    *   The Test Set consists of entirely unseen EoS curves.

3.  **Hyperparameter Configuration**
    Two distinct configurations are defined for the Random Forest algorithm:
    *   **Observational Config (Models Geo, A):** Uses "Bagging" (training on random subsets of data) and allows trees to consider all features at every split. This is necessary because Mass and Radius are strongly correlated, and the model must see both to resolve ambiguities at low masses.
    *   **Physics Config (Models B, C, D):** Forces the trees to consider only a subset of features ($\sqrt{N_{features}}$) at each split. This "decorrelation" forces the model to learn from subtle microphysical features (like Slope or Sound Speed) rather than relying solely on Radius.

4.  **Training and Calibration Loop**
    For each of the five models:
    *   The specific feature subset is selected.
    *   The Random Forest is trained on the training fold.
    *   The model is calibrated using Cross-Validation on the training data.
    *   The final calibrated model is stored.

5.  **Performance Diagnostics**
    The accuracy is evaluated on both the Training and Test sets.
    *   **Overfitting Check:** If the Training Accuracy is significantly higher ($> 5\%$) than the Test Accuracy, the model is flagged as overfit.
    *   **Underfitting Check:** If the Training Accuracy is low ($< 70\%$), the model is flagged as underfit.

## Inputs and Outputs

### Inputs
*   **Dataset:** A pandas DataFrame containing the balanced, shuffled set of Hadronic and Quark stars, with all computed physical features.

### Outputs
*   **Models Dictionary:** A collection of trained, calibrated classifiers (`Geo`, `A`, `B`, `C`, `D`).
*   **Test Data ($X_{test}, y_{test}$):** The subset of data held out for final evaluation. This ensures that all subsequent plots (ROC curves, Confusion Matrices) are generated from data the models have never seen.

## Connection to the Overall Workflow
This script transforms the raw physical simulations into predictive tools.
*   It takes the data generated by the **Physics Engine**.
*   It produces the models used by the **Visualization Suite** (e.g., for Partial Dependence Plots and ROC curves).
*   It provides the specific models (`Geo` and `A`) used in **Inference** to classify real astrophysical objects like GW170817.
