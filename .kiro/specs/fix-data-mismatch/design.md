# Design Document

## Overview

This design addresses critical data mismatches that cause poor simulation results:
1. Station ID mismatch (data uses 0-based, environment uses 1-based)
2. Passengers arriving outside service hours
3. Insufficient capacity validation

The solution involves modifying the data loading pipeline to transform station IDs and filter out-of-service passengers.

## Architecture

### Component Interaction

```
PassengerDataLoader → StationLevelBusEnvironment
         ↓                      ↓
   Transform IDs          Validate & Load
         ↓                      ↓
   Filter Times           Assign to Stations
```

### Key Changes

1. **Data Transformation Layer**: Add station ID conversion (+1) during data loading
2. **Time Filtering**: Filter passengers outside service window
3. **Validation Layer**: Verify data integrity after loading
4. **Capacity Analysis**: Pre-check if demand exceeds capacity

## Components and Interfaces

### 1. StationLevelBusEnvironment Modifications

**Method: `load_passenger_data`**
- Add `filter_out_of_service` parameter (default: True)
- Add station ID conversion logic
- Add validation logging

**Method: `_process_passenger_dataframe`**
- Convert station IDs: `boarding_station + 1`, `alighting_station + 1`
- Filter by time: `service_start <= boarding_time < service_end`
- Log filtered passenger counts

**New Method: `validate_passenger_loading`**
```python
def validate_passenger_loading(self, expected_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Validate that passengers were loaded correctly
    
    Returns:
        {
            'up_loaded': int,
            'down_loaded': int,
            'up_expected': int,
            'down_expected': int,
            'match': bool,
            'issues': List[str]
        }
    """
```

### 2. Capacity Analysis Utility

**New Function: `analyze_capacity_requirements`**
```python
def analyze_capacity_requirements(
    passenger_count: int,
    bus_capacity: int,
    service_duration: int,
    tmin: int,
    tmax: int
) -> Dict[str, Any]:
    """
    Analyze if capacity is sufficient
    
    Returns:
        {
            'passenger_count': int,
            'theoretical_max_departures': int,  # service_duration / tmin
            'theoretical_min_departures': int,  # service_duration / tmax
            'required_departures': int,  # ceil(passenger_count / bus_capacity)
            'is_feasible': bool,
            'recommendations': List[str]
        }
    """
```

### 3. Data Validation Script

**New Script: `validate_data_loading.py`**
- Load data and environment
- Compare counts
- Check station ID ranges
- Check time ranges
- Report discrepancies

## Data Models

### Passenger (Modified)

```python
@dataclass
class Passenger:
    passenger_id: int
    arrival_time: int  # Must be in [service_start, service_end)
    boarding_station: int  # Converted to 1-based (1 to num_stations)
    alighting_station: int  # Converted to 1-based (1 to num_stations)
    direction: str  # 'up' or 'down'
    boarding_time: Optional[int] = None
```

### Validation Result

```python
@dataclass
class ValidationResult:
    passengers_loaded: int
    passengers_expected: int
    passengers_filtered: int
    station_id_min: int
    station_id_max: int
    time_min: int
    time_max: int
    issues: List[str]
    is_valid: bool
```

## Error Handling

### Station ID Out of Range

```python
if converted_station_id < 1 or converted_station_id > num_stations:
    logger.warning(f"Invalid station ID {converted_station_id} after conversion")
    continue  # Skip this passenger
```

### Time Out of Range

```python
if arrival_time < service_start or arrival_time >= service_end:
    filtered_count += 1
    if filter_out_of_service:
        continue  # Skip this passenger
```

### Capacity Insufficient

```python
if required_departures > max_possible_departures:
    logger.error(f"Cannot serve all passengers: need {required_departures} but max is {max_possible_departures}")
    # Provide recommendations
```

## Testing Strategy

### Unit Tests

1. **Test Station ID Conversion**
   - Input: passenger with boarding_station=0
   - Expected: Passenger assigned to station_id=1

2. **Test Time Filtering**
   - Input: passengers at times [300, 360, 1260, 1300]
   - Expected: Only [360] loaded (if service is 360-1260)

3. **Test Capacity Calculation**
   - Input: 5000 passengers, 48 capacity, 900 minutes, tmin=3
   - Expected: required=105 departures, max=300, feasible=True

### Integration Tests

1. **Test Full Data Loading**
   - Load real 208 route data
   - Verify all in-service passengers loaded
   - Verify station IDs in valid range
   - Verify no passengers outside service window

2. **Test Environment Simulation**
   - Run one episode
   - Verify passengers arrive at stations
   - Verify buses pick up passengers
   - Verify stranded count is reasonable

### Validation Tests

1. **Compare Before/After**
   - Before fix: High stranded passengers (600+)
   - After fix: Low stranded passengers (<50)

2. **Verify Waiting Times**
   - Before fix: 100+ minutes average
   - After fix: <10 minutes average (closer to paper's 3.7-3.8)

## Implementation Notes

### Station ID Conversion

The conversion must happen during `_process_passenger_dataframe`:

```python
boarding_station=int(row['Boarding station']) + 1,  # Convert 0-based to 1-based
alighting_station=int(row['Alighting station']) + 1,  # Convert 0-based to 1-based
```

### Time Filtering

Add filtering logic before creating Passenger objects:

```python
arrival_time = int(row['Boarding time'])
if filter_out_of_service:
    if arrival_time < self.service_start or arrival_time >= self.service_end:
        filtered_count += 1
        continue
```

### Backward Compatibility

To maintain compatibility with existing code:
- Add `filter_out_of_service` parameter with default=True
- Add `convert_station_ids` parameter with default=True
- Log all transformations for transparency

## Performance Considerations

- Station ID conversion: O(n) where n = number of passengers
- Time filtering: O(n) where n = number of passengers
- Overall impact: Negligible (< 1ms for 5000 passengers)

## Deployment Plan

1. **Phase 1**: Implement station ID conversion
2. **Phase 2**: Implement time filtering
3. **Phase 3**: Add validation utilities
4. **Phase 4**: Run full training and compare results
5. **Phase 5**: Document improvements in results

## Success Metrics

- Stranded passengers: Target < 50 (currently 600+)
- Average waiting time: Target < 10 minutes (currently 100+)
- Passengers served: Target > 95% (currently ~60%)
- Training convergence: Faster and more stable
