# Preference-to-Cost Translator for Route Optimization

ML system that translates human preferences into ODL optimizer cost parameters.

## Problem

Route optimization requires setting 10-30 cost parameters (e.g., `costPerKm`, `costParkingDifficulty`). Currently manual trial-and-error. 

## Solution

**Input:** Natural preferences
```python
preferences = {
    "parking_importance": 0.85,  # Avoid difficult parking
    "time_importance": 0.60,      # Moderately care about time
    "distance_importance": 0.30   # Flexible on distance
}
```

**Output:** Optimized cost parameters
```python
odl_costs = {
    "costPerKm": 8.5,
    "costPerHour": 450.0,
    "parkingMultiplier": 2.5,
    # ... other parameters
}
```

## Architecture

1. **Translator**: ML model (regression/inverse optimization) maps preferences → costs
2. **Validator**: Checks if generated routes match intended preferences  
3. **Feedback Loop**: Iteratively improves translation

## Data

- 106 delivery jobs in Oslo
- 26 vehicles
- Binary parking difficulty (easy=0, difficult=500)
- Real addresses and time windows

## Status

🚧 Research project - Data analysis complete, ML implementation in progress

## Tech Stack

Python · scikit-learn · XGBoost · PyTorch · scipy.optimize
