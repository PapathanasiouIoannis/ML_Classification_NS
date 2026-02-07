# Hyperparameter Optimization (Grid Search)

## Role in the Project
This script is the **optimization engine** for the machine learning pipeline. Its purpose is to systematically search for the most effective configuration of the Random Forest classifiers, ensuring that the final models achieve maximum accuracy without overfitting to the training data.

It acknowledges that different input feature sets require different learning strategies. Therefore, it conducts separate optimization routines for:
1.  **Observational Models (Model A/Geo):** Which rely on limited, highly correlated features (Mass and Radius).
2.  **Physics Models (Model D):** Which have access to rich, decorrelated microphysical features (Energy Density, Sound Speed, Slope).

## Physics and Equations

### 1. The Bias-Variance Tradeoff
The optimization process navigates the Bias-Variance tradeoff to minimize the Generalization Error $E_{gen}$. The total error can be decomposed as:

$$
E_{gen} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

*   **Bias:** Error due to erroneous assumptions (e.g., a tree that is too shallow to capture the complex EoS topology).
*   **Variance:** Error due to sensitivity to small fluctuations in the training set (e.g., a tree that is too deep and memorizes noise).

The hyperparameters tuned here control this balance:
*   `max_depth`: Limits the complexity of the decision boundary (reducing Variance).
*   `min_samples_leaf`: Enforces smoothness by requiring a minimum number of stars in each leaf node.
*   `max_features`: Controls the correlation between trees in the forest.

### 2. Feature Correlation Strategy
A critical physical insight drives the tuning strategy:
*   **Mass and Radius are strongly correlated.** In a typical neutron star sequence, determining the mass strongly constrains the radius. If the model is forced to pick only one feature at a split (`max_features='sqrt'`), it might lose context. Therefore, Observational models are tested with `max_features=None` (seeing all features).
*   **Microphysics features are distinct.** Central density $\epsilon_c$ and sound speed $c_s^2$ provide orthogonal information to the macroscopic observables. Physics models benefit from `max_features='sqrt'` to force trees to learn these independent physical drivers.

## Algorithm and Calculations

1.  **Data Preparation**
    *   The dataset is loaded and cleaned.
    *   Logarithmic transformations are applied to the Tidal Deformability ($\Lambda$) to stabilize its wide dynamic range.
    *   Stars are grouped by their `Curve_ID`. This is crucial for the cross-validation strategy.

2.  **Group K-Fold Cross-Validation**
    Standard random splitting is scientifically invalid here because stars from the same Equation of State (EoS) curve are physically related. If stars from Curve X are in both the training and validation sets, the model will "memorize" Curve X rather than learning general physics.
    To prevent this leakage, a **Group K-Fold** splitter is used:
    *   The dataset is divided into $K=3$ folds.
    *   **Constraint:** All stars belonging to a specific EoS curve must appear in the same fold.
    *   The model is trained on $K-1$ folds and validated on the remaining fold.

3.  **Grid Search Execution**
    Two distinct grid searches are performed:

    *   **Scenario A (Observational):**
        *   Features: Mass, Radius, $\log\Lambda$.
        *   Hypothesis: The model needs to see all features simultaneously to resolve the $M-R$ degeneracy.
        *   Grid: Tests `max_features` options (`sqrt` vs `None`) and regularization depths.

    *   **Scenario D (Microphysics):**
        *   Features: Mass, Radius, $\log\Lambda$, $\epsilon_c$, $c_s^2$, Slope.
        *   Hypothesis: The model should be forced to decorrelate trees to exploit the rich feature set.
        *   Constraint: `max_features` is locked to `'sqrt'`.

4.  **Metric Evaluation**
    For every combination of hyperparameters, the average validation accuracy across the 3 folds is computed. The set of parameters that yields the highest accuracy is reported as the "Best Params."

## Inputs and Outputs

### Inputs
*   **Dataset:** The `thesis_dataset.csv` file containing the full population of simulated stars.

### Outputs
*   **Optimization Report:** A printed summary to the console detailing:
    *   The best hyperparameter set for Model A.
    *   The best hyperparameter set for Model D.
    *   The maximum cross-validation accuracy achieved for each.

## Connection to the Overall Workflow
This script is usually run **offline** or **intermittently**. It does not need to run every time the pipeline executes.

Its results (the "Best Params") are manually transferred to `train_model.py` to hardcode the configuration of the final classifiers. This ensures that the production models are always using the scientifically optimal settings derived from this rigorous search.
