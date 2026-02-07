# Model Performance Audit and Stress Testing

## Role in the Project
This script executes a rigorous "stress test" on the trained machine learning models. While global metrics like overall accuracy provide a general summary of performance, they often mask localized failures in specific regions of the parameter space.

The primary role of this module is to dissect the model's behavior as a function of stellar mass and to evaluate the "honesty" of its uncertainty estimates. It answers three critical questions:
1.  Does the model's reliability degrade for very massive or very light stars?
2.  When the model makes a mistake, is it "confidently wrong" or appropriately uncertain?
3.  How robust is the classification in the extreme high-mass regime ($M > 2.0 M_{\odot}$), which is critical for current astrophysical constraints?

## Physics and Equations

### 1. Mass-Dependent Accuracy
Neutron star properties change drastically with mass. Low-mass stars are dominated by the known nuclear crust, while high-mass stars are dominated by the unknown core Equation of State (EoS). To quantify performance across this range, the test set is divided into mass bins $B_i$:

$$
B_i = \{ \mathbf{x} \in X_{test} \mid M_{low, i} \le M < M_{high, i} \}
$$

For each bin, the local classification accuracy is computed:

$$
\text{Accuracy}(M) = \frac{1}{|B_i|} \sum_{\mathbf{x} \in B_i} \mathbb{I}(\hat{y}(\mathbf{x}) = y_{true})
$$

where $\mathbb{I}$ is the indicator function (1 if correct, 0 if incorrect). This reveals the "Zone of Confidence"—the mass range where the model is most reliable.

### 2. Confidence of Errors (Calibration Check)
A well-calibrated model should output probabilities near $0.5$ (uncertainty) when it is likely to be incorrect. If a model outputs a probability $P \approx 1.0$ (certainty) but the prediction is wrong, it is "hallucinating."

The confidence level $C$ is defined as the distance of the predicted probability $P$ from the decision boundary ($0.5$), normalized to $[0, 1]$:

$$
C = 2 \times | P(\text{Quark}) - 0.5 |
$$

The distribution of $C$ is analyzed specifically for the subset of misclassified stars. A distribution skewed toward $0$ indicates a healthy, honest model; a distribution skewed toward $1$ indicates dangerous overfitting.

### 3. The High-Mass Limit
The behavior of the EoS at high densities is of particular interest to nuclear physics. Massive stars ($M > 2.0 M_{\odot}$) probe the deepest central densities. A specific accuracy metric is computed for this subset:

$$
\text{Accuracy}_{high} = \text{Accuracy} \quad \text{for} \quad \{ \mathbf{x} \mid M > 2.0 M_{\odot} \}
$$

This ensures that the model does not achieve high global accuracy simply by correctly classifying the abundant low-mass stars while failing on the rare, scientifically valuable high-mass cases.

## Algorithm and Calculations

1.  **Mass Binning Analysis:**
    *   The mass spectrum (e.g., $0.1$ to $3.0 M_{\odot}$) is segmented into discrete intervals.
    *   The test dataset is filtered to isolate stars falling within each interval.
    *   For each model (Geometric, Observational, Microphysical), predictions are generated for the stars in the current bin.
    *   The accuracy is calculated and plotted against the bin center.
    *   A sample count histogram is generated in parallel to ensure that bins with low accuracy are statistically significant and not just artifacts of small sample sizes.

2.  **Error Confidence Analysis:**
    *   The Observational Model (Model A) is selected for detailed scrutiny.
    *   Predictions are compared against true labels to create a mask of **Misclassified Samples**.
    *   The predicted probabilities associated with these errors are extracted.
    *   A histogram of these probabilities is plotted. Ideally, the histogram peaks around $0.5$, indicating that the model was "unsure" about the cases it got wrong.

3.  **High-Mass Critical Check:**
    *   A subset of the test data is created containing only stars with masses greater than $2.0 M_{\odot}$.
    *   Each model is scored against this subset.
    *   A status flag is assigned based on the score (e.g., "Critical Failure" if accuracy drops below 60%, "Robust" if above 90%).

## Inputs and Outputs

### Inputs
*   **Models Dictionary:** The collection of trained classifiers (Geo, A, D).
*   **Test Data ($X_{test}, y_{test}$):** The held-out dataset used for evaluation.

### Outputs
*   **Accuracy vs. Mass Plot:** A dual-axis figure showing how accuracy evolves with stellar mass, overlaid with the distribution of data points.
*   **Error Confidence Histogram:** A visualization of the model's internal probability estimates for incorrect predictions.
*   **Audit Report:** A console summary detailing the robustness of the models in the high-mass regime.

## Connection to the Overall Workflow
This script serves as the **Quality Assurance** phase of the pipeline. It is executed immediately after `train_model.py`.

The results from this audit determine whether the models are trustworthy enough to be applied to real observational data in `analyze_candidates.py`. If a model fails the High-Mass check here, its predictions on objects like PSR J0740+66 would be considered unreliable.
