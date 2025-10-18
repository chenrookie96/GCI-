# Implementation Plan

- [x] 1. Implement station ID conversion in data loading


  - Modify `_process_passenger_dataframe` to add 1 to all station IDs
  - Add validation that converted IDs are within valid range [1, num_stations]
  - Log conversion statistics (min/max IDs before and after)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_


- [ ] 2. Implement time filtering for out-of-service passengers
  - Add `filter_out_of_service` parameter to `load_passenger_data` method
  - Filter passengers with arrival_time < service_start
  - Filter passengers with arrival_time >= service_end
  - Log count of filtered passengers for each direction

  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 3. Add passenger loading validation
  - Create `validate_passenger_loading` method in StationLevelBusEnvironment
  - Compare loaded passenger count with expected count from source data
  - Validate all station IDs are in valid range
  - Validate all arrival times are within service window
  - Return detailed validation report with any discrepancies
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 4. Create capacity analysis utility
  - Create new file `src/utils/capacity_analysis.py`
  - Implement `analyze_capacity_requirements` function
  - Calculate theoretical max/min departures based on constraints
  - Calculate required departures based on passenger demand


  - Provide feasibility check and recommendations
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 5. Create data validation script
  - Create `validate_data_loading.py` script
  - Load passenger data and create environment
  - Run validation checks
  - Display before/after comparison of station IDs
  - Display time range analysis
  - Display capacity analysis
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 6. Update environment passenger tracking
  - Verify `_update_passenger_arrivals` correctly adds passengers to stations
  - Verify `_update_buses` correctly boards passengers from stations


  - Add detailed logging for passenger boarding events
  - Track total_passengers_served counter
  - Calculate stranded_passengers at end of service
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 7. Run validation and verify fixes
  - Run `validate_data_loading.py` to confirm fixes
  - Verify all passengers have correct station IDs (1-based)
  - Verify all passengers are within service window
  - Verify capacity is sufficient for passenger demand
  - Document validation results
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 8. Re-run training with fixed data
  - Run training script with corrected data loading
  - Monitor stranded passenger counts (target < 50)
  - Monitor average waiting times (target < 10 minutes)
  - Compare results with paper benchmarks
  - Document improvements in training results
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5_
