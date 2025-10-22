# ODL Test Model Data Analysis

## Executive Summary

This is a **real ODL (route optimization) model configuration** for Oslo delivery operations containing:
- **106 delivery jobs** (pickup + delivery pairs)
- **26 vehicles** available for routing
- Complete cost parameters, time windows, and constraints
- Real Oslo addresses and coordinates

This data is exactly what your Preference-to-Cost Translator system will work with!

---

## Data Structure Overview

```
test_model_mathias.json
├── lastModifiedTimestamp: "2025-10-22T20:08:22.01"
├── _id: "v2_49"
├── configuration: {...}  # ODL optimizer settings
└── data:
    ├── jobs: [...]       # 106 delivery jobs
    └── vehicles: [...]   # 26 available vehicles
```

---

## 1. JOBS Section (Delivery Orders)

### Structure
Each job represents a **shipment** with two stops:
1. **PICKUP** - where goods are collected
2. **DELIVERY** - where goods are delivered to customer

### Example Job
```json
{
  "_id": "3935496",
  "requiredSkills": ["alle", "driver-305"],
  "quantities": [0, 0, 0],
  "stops": [
    {
      "type": "SHIPMENT_PICKUP",
      "_id": "7868353",
      "address": "Maridalsveien 300, 0872 OSLO",
      "coordinate": {
        "latitude": 59.953677,
        "longitude": 10.760444
      },
      "durationMillis": 1,
      "parking": {
        "parkingTimeSeconds": 600.0,
        "cost": 0.0  // ← PICKUP: No parking cost
      },
      "multiTWs": [{
        "openTime": "2025-10-22T04:00:00",
        "closeTime": "2025-10-22T20:00:00",
        "penalties": [
          {
            "cost": 0.0,
            "costPerHour": 0.0,
            "openTime": "2025-10-22T04:00:00"
          },
          {
            "cost": 1.0,
            "costPerHour": 1.0,
            "costPerHourSqd": 5.0,  // ← Penalty increases after 4pm
            "openTime": "2025-10-22T16:00:00"
          }
        ]
      }]
    },
    {
      "type": "SHIPMENT_DELIVERY",
      "_id": "7868354",
      "address": "Karl Johans gate 5, 0154 OSLO",
      "coordinate": {
        "latitude": 59.911582,
        "longitude": 10.748199
      },
      "durationMillis": 1080000,  // 18 minutes
      "parking": {
        "parkingTimeSeconds": 600.0,
        "cost": 500.0  // ← DELIVERY: High parking difficulty cost!
      },
      "multiTWs": [{
        "openTime": "2025-10-22T06:00:00",
        "closeTime": "2025-10-22T17:00:00",
        "penalties": [...]
      }]
    }
  ]
}
```

### Key Job Features

#### Parking Costs
- **Range**: 0 to 500
- **Interpretation**: 
  - `0` = Easy parking (residential areas)
  - `500` = Difficult parking (city center, Karl Johans gate)
- **THIS IS CRITICAL FOR YOUR PROJECT**: The `parking.cost` values are exactly what your preferences need to influence!

#### Time Windows (multiTWs)
- Each stop can have multiple time windows with penalties
- Penalties increase for late arrivals (quadratic cost)
- Example: Prefer before 1pm, but allow until 5pm with penalty

#### Skills Required
- `"alle"` = General requirement
- `"driver-305"` = Specific driver constraint

---

## 2. VEHICLES Section (Fleet)

### Structure
Each vehicle has a definition with cost parameters and capabilities.

### Example Vehicle Cost Parameters
```json
{
  "_id": "754069",
  "definition": {
    "costPerTravelHour": 450.0,     // ← Cost per hour driving
    "costPerKm": 8.5,                // ← Cost per kilometer
    "costPerWaitingHour": 200.0,    // ← Cost waiting at stops
    "costPerServicingHour": 300.0,  // ← Cost during service
    "costFixed": 1000.0,             // ← Fixed cost to use vehicle
    "costPerStop": 50.0,             // ← Cost per stop visited
    
    "capacities": [100, 100, 100],
    "skills": ["alle"],
    
    "start": {
      "address": "Lørenveien 40, 0585 OSLO",
      "coordinate": {...},
      "openTime": "2025-10-22T06:00:00"
    },
    "end": {
      "address": "Lørenveien 40, 0585 OSLO", 
      "closeTime": "2025-10-22T18:00:00"
    }
  }
}
```

### **THIS IS THE KEY**: Vehicle Cost Parameters

These are the **exact parameters your ML model needs to learn to set**:

| Parameter | Typical Value | What It Controls |
|-----------|---------------|------------------|
| `costPerKm` | 8.5 | Route distance preference |
| `costPerTravelHour` | 450.0 | Time importance |
| `costPerStop` | 50.0 | Stop consolidation |
| `costPerWaitingHour` | 200.0 | Early arrival penalty |
| `costFixed` | 1000.0 | Vehicle utilization |

**Note**: Parking difficulty costs are in the **job stops**, not vehicle definition!

---

## 3. CONFIGURATION Section (Optimizer Settings)

### Optimizer Configuration
```json
{
  "optimiser": {
    "alternatePlans": [{
      "maxIterations": 9999,
      "maxMillisecondsRuntime": 600000,  // 10 minutes max
      "maxJobsPerSplit": 500
    }],
    "recursiveSplits": {
      "enabled": true,
      "maxJobsPerSplit": 500
    },
    "engineVersion": 3
  },
  "problem": {
    "latenessPenalty": {
      "power": 2.0,
      "multiplier": 99999.0  // ← Very high penalty for late delivery
    }
  }
}
```

---

## 4. Geographic Analysis

All 106 jobs are in **Oslo**, with addresses like:
- `Maridalsveien 300, 0872 OSLO`
- `Karl Johans gate 5, 0154 OSLO` (city center - high parking cost!)
- `Støperigata 2, 0250 OSLO`

This matches your proposal's **"oslo_centrum"** geographic zone!

---

## 5. What This Means For Your Project

### You Have Real Training Data!

#### Features You Can Extract:

**From Jobs:**
- Parking difficulty distribution (0-500 range)
- Time window flexibility (hard vs soft)
- Geographic clustering (coordinates)
- Delivery duration variability

**From Vehicles:**
- Current cost parameter configurations
- Vehicle capacity utilization
- Operating time windows

### Mapping to Your Proposal

Your proposed preference structure:
```python
preferences = {
    "parking_importance": 0.85,
    "distance_importance": 0.30,
    "time_importance": 0.60,
}
```

Should map to these **ODL cost parameters**:
```python
odl_costs = {
    "cost_per_km": 8.5,              # ← controls distance_importance
    "cost_per_hour": 450.0,          # ← controls time_importance
    "cost_parking_difficulty": ???   # ← This might be applied as multiplier
}
```

### Critical Insight
The `parking.cost` in stops is **already set** (0 or 500). Your system needs to learn how to **weight these existing costs** through vehicle parameters or preference multipliers!

---

## 6. Data Quality Assessment

### ✅ What's Good:
- Real addresses with coordinates
- Realistic parking costs (0 for easy, 500 for difficult)
- Realistic time windows (6am-6pm typical)
- Actual Oslo geography
- Multiple vehicles with different configurations

### ⚠️ What's Missing (for your project):
- **Historical route acceptance/rejection data** (needed for inverse optimization)
- **Actual driver preferences** (need to be collected or simulated)
- **Route outcomes** (which routes were actually used?)
- **Feedback data** (driver complaints about parking, etc.)

---

## 7. Next Steps for Your Project

### Immediate Actions:

1. **Understand Current Costs**: 
   - Run this model through ODL API
   - See what routes it generates
   - Analyze if they prefer easy parking (cost 0) vs difficult (cost 500)

2. **Create Preference Scenarios**:
   ```python
   scenario_1 = {
       "parking_importance": 0.9,  # Strongly avoid difficult parking
       "time_importance": 0.3       # Flexible on time
   }
   
   scenario_2 = {
       "parking_importance": 0.2,  # Don't care about parking
       "time_importance": 0.9       # Minimize time
   }
   ```

3. **Experiment with Cost Adjustments**:
   - Try increasing `costPerTravelHour` → should reduce total distance
   - Try multiplying parking costs → should avoid high-cost stops
   - Document which cost changes achieve which preference outcomes

4. **Build Baseline Validator**:
   - Run model with current costs
   - Count stops with parking cost > 250
   - Calculate total distance, time, parking difficulty score

### Questions to Answer:

1. **How do parking costs currently influence routing?**
   - Does the optimizer naturally avoid cost=500 stops?
   - Or does it need additional weighting?

2. **Where are the cost parameters stored?**
   - Are they in vehicle definitions? ✓
   - Are they global? (need to check configuration)
   - Can they be stop-specific?

3. **Can you get actual route outputs from this model?**
   - This would let you validate preference matching
   - You could create training data

---

## Summary

You have a **production-ready ODL model** with:
- ✅ 106 real delivery jobs in Oslo
- ✅ Parking difficulty indicators (0 vs 500)
- ✅ Vehicle cost parameters to optimize
- ✅ Time windows and constraints
- ❌ No historical preference/outcome data (yet)

**Your translator system needs to learn**: Given a preference like "parking_importance: 0.85", what vehicle cost parameters will make ODL generate routes that actually avoid difficult parking?

This is the perfect dataset to start prototyping!