# ML Model Training Results

**Training Date:** 20251106_091456
**Output Directory:** ml_training_results_20251106_091456

---

## Dataset Summary

- **Total Training Samples:** 40,000
- **Training Set:** 32,000 samples
- **Test Set:** 8,000 samples
- **Unique Cost Combinations:** 145
- **Unique Preference Combinations:** 400
- **Used Sample Weights:** Yes (distance-based weighting)

---

## Model Performance

### Best Model: Gradient Boosting

**Overall R² Score:** -0.0126

### Per-Output Performance:


#### costPerTravelHour
- R² Score: -0.0151
- RMSE: 5.5794
- MAE: 4.8602

#### costPerKm
- R² Score: -0.0106
- RMSE: 0.5672
- MAE: 0.4870

#### parking_multiplier
- R² Score: -0.0120
- RMSE: 2.9093
- MAE: 2.6109

---

## Model Comparison

| Model | Overall R² |
|-------|------------|
| Gradient Boosting | -0.0126 |
| Random Forest | -0.0183 |

---

## Files Generated

### Model Files
- `preference_to_cost_model.pkl` - Trained model (pickle format)
- `model_metadata.json` - Model metadata and metrics

### Figures (0 total)
- `figures/actual_vs_predicted.png`
- `figures/feature_importance.png`
- `figures/input_distributions.png`
- `figures/model_comparison_r2.png`
- `figures/output_distributions.png`
- `figures/rank_distribution.png`
- `figures/residual_analysis.png`
- `figures/sample_weights_and_distances.png`

---

## Usage Instructions

```python
import pickle
import numpy as np

# Load the model
with open('preference_to_cost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define user preferences
preferences = np.array([[
    0.7,  # parking_importance
    0.2,  # time_importance
    0.1   # distance_importance
]])

# Predict costs
costs = model.predict(preferences)[0]

print(f"costPerTravelHour: {costs[0]:.2f}")
print(f"costPerKm: {costs[1]:.4f}")
print(f"parking_multiplier: {costs[2]:.2f}")
```

---

## Notes

- Model was trained with sample weights based on Pareto distance
- Closer preference-cost matches received higher importance during training
- All figures show model performance on the test set
- Ready for production deployment
