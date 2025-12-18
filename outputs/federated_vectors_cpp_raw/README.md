# Federated Raw Data Vectors (C++)

This directory contains per-city datasets exported as **raw numerical data vectors** for use in a C++-based federated learning framework.

Each city is treated as an independent client and includes locally split training, validation, and test sets.

---

## Directory Structure

For each city (e.g., `nyc/`, `philadelphia/`):

<city>/
X_train.bin
y_train_raw.bin
X_val.bin
y_val_raw.bin
X_test.bin
y_test_raw.bin
meta.json

-----

## Feature Vectors

All feature matrices (`X_*.bin`) are stored as **row-major float32** arrays with a fixed column order:

[ site_eui, energy_star_score, gross_floor_area_sqft ]


The column order is critical and is also recorded in `meta.json`.

Each `X_*.bin` file represents a matrix of shape:


---

## Target Values (Raw Scale)

Targets (`y_*_raw.bin`) are stored on the **original emissions scale**:


Each target file is a **float32 vector** of length `n_samples`.

Raw targets are provided for interpretability and to avoid imposing a specific transformation inside the federated framework.

---

## Optional: Log-Transforming the Target

Greenhouse gas emissions are heavily right-skewed.  
If you wish to train in log space (as done in the centralized baseline), you can apply a `log1p` transform to the raw target:

y_log = log(1 + y_raw)

Implementation of log transform below

```cpp
#include <cmath>
#include <vector>
#include <cstddef>

// Convert raw emissions to log space
std::vector<float> log1p_target(const std::vector<float>& y_raw) {
    std::vector<float> y_log(y_raw.size());
    for (size_t i = 0; i < y_raw.size(); i++) {
        float v = y_raw[i];
        if (v < 0.0f) v = 0.0f;   // guard against invalid values
        y_log[i] = std::log1p(v);
    }
    return y_log;
}

// Convert predictions back to raw scale if needed
float expm1_pred(float y_log_pred) {
    return std::expm1(y_log_pred);
}




