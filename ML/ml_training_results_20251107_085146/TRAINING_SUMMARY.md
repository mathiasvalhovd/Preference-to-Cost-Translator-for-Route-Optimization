# ML Model Training Results

**Training Date:** 20251107_085146
**Output Directory:** ml_training_results_20251107_085146

---

## Dataset Summary

- **Total Training Samples:** 400
- **Training Set:** 320 samples
- **Test Set:** 80 samples
- **Unique Cost Combinations:** 5
- **Unique Preference Combinations:** 400
- **Used Sample Weights:** Yes (distance-based weighting)

---

## Model Performance

### Best Model: Gradient Boosting

**Overall R² Score:** 0.6483

### Per-Output Performance:


#### costPerTravelHour
- R² Score: 0.6016
- RMSE: 2.5716
- MAE: 0.4877

#### costPerKm
- R² Score: 0.4519
- RMSE: 0.2240
- MAE: 0.0376

#### parking_multiplier
- R² Score: 0.8914
- RMSE: 0.0285
- MAE: 0.0071

---

## Model Comparison

| Model | Overall R² |
|-------|------------|
| Gradient Boosting | 0.6483 |
| Random Forest | 0.6318 |

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
