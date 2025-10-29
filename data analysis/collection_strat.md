# For each sample:
1. Generate random preferences (3 values between 0-1)
2. Generate random costs (based on preference ranges)
3. Send to ODL API with costs applied
4. Wait ~2-3 minutes for optimization
5. Get routes back
6. Calculate route features (parking score, distance, time)
7. Save: (preferences, costs, route_features) as one training sample


## All costs??
Vehicle-Level Costs (in vehicle.definition):

costPerTravelHour - Cost per hour of driving (currently: 1.0)
costPerKm - Cost per kilometer traveled (currently: 0.028)
costPerWaitingHour - Cost per hour of waiting (currently: 1.0)
costPerServicingHour - Cost per hour at stops (currently: 1.0)
costFixed - Fixed cost for using the vehicle (currently: 0.0)
costPerStop - Cost per stop visited (currently: 0.0)

Stop-Level Costs (in job.stops[].parking):

parking.cost - Parking difficulty cost (0 or 500 in your data)

Penalty Costs (in time windows):

Time window penalties - cost, costPerHour, costPerHourSqd for being late