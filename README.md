# StatCheck

A clean, interactive Streamlit web app for performing hypothesis tests on means using Z-tests and T-tests.  
The app supports one-sample and two-sample (independent) mean tests and automatically selects the appropriate test statistic based on sample size.

---

## Features

- One-sample mean hypothesis testing
- Two-sample mean hypothesis testing (independent samples only, Welch method)
- Automatic Z vs. T selection:
  - Uses **Z** when *both* sample sizes are greater than 40
  - Uses **T** otherwise
- User-selectable significance level (α)
- Supports two-sided, left-tailed, and right-tailed alternatives
- Displays:
  - Null and alternative hypotheses
  - Rejection region
  - Test statistic and p-value
  - Clear statistical conclusion
- Clean, minimal UI designed for clarity and classroom use

---

## How the App Works

### Test Selection
Users choose between:
- **1 Mean (one sample)**  
- **2 Means (independent samples)**

### Inputs
Depending on the test, users provide:
- Significance level (α)
- Alternative hypothesis
- Hypothesized mean (μ₀) or hypothesized difference (Δ₀)
- One or two samples of numeric data (entered as lists)

### Statistical Method
- **1-sample tests** use:
  - Z-test if n > 40
  - T-test if n ≤ 40
- **2-sample tests** use:
  - Welch’s test for unequal variances
  - Z-test only if both sample sizes exceed 40

Sample standard deviation is used as an approximation for population standard deviation in Z-tests.