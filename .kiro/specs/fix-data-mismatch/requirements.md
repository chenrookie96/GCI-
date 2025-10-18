# Requirements Document

## Introduction

This feature addresses critical data mismatches between the passenger data and the simulation environment that are causing poor training results. The system currently has station ID mismatches and passengers arriving outside service hours, leading to high stranded passenger counts and unrealistic waiting times.

## Glossary

- **System**: The DRL-TSBC bus scheduling simulation system
- **Passenger_Data**: Real-world passenger boarding records from test_data
- **Environment**: The StationLevelBusEnvironment simulation
- **Station_ID**: Unique identifier for bus stops (0-based in data, 1-based in environment)
- **Service_Window**: The operational time period from 06:00 to 21:00
- **Stranded_Passenger**: A passenger who cannot board any bus during service hours

## Requirements

### Requirement 1

**User Story:** As a researcher, I want the passenger data station IDs to match the environment station IDs, so that passengers are correctly assigned to stations

#### Acceptance Criteria

1. WHEN THE System loads passenger data, THE System SHALL convert station IDs from 0-based to 1-based indexing
2. WHEN a passenger has boarding_station=0 in data, THE System SHALL assign the passenger to station_id=1 in Environment
3. WHEN a passenger has boarding_station=N in data, THE System SHALL assign the passenger to station_id=N+1 in Environment
4. WHEN THE System processes alighting stations, THE System SHALL apply the same ID conversion (add 1)
5. THE System SHALL validate that all converted station IDs fall within the valid range of 1 to num_stations

### Requirement 2

**User Story:** As a researcher, I want passengers arriving outside service hours to be filtered or adjusted, so that the simulation only handles serviceable passengers

#### Acceptance Criteria

1. WHEN THE System loads passenger data, THE System SHALL identify passengers with arrival_time < service_start
2. WHEN THE System loads passenger data, THE System SHALL identify passengers with arrival_time >= service_end
3. THE System SHALL provide configuration option to filter out-of-service passengers
4. THE System SHALL log the count of filtered passengers for each direction
5. WHEN filtering is enabled, THE System SHALL exclude passengers outside Service_Window from simulation

### Requirement 3

**User Story:** As a researcher, I want to verify that all passengers can be served with the available capacity, so that I can ensure the simulation is realistic

#### Acceptance Criteria

1. THE System SHALL calculate total_passenger_demand for each direction
2. THE System SHALL calculate theoretical_max_capacity as departures × bus_capacity
3. WHEN total_passenger_demand > theoretical_max_capacity, THE System SHALL log a capacity warning
4. THE System SHALL report capacity_utilization_rate as total_passenger_demand / theoretical_max_capacity
5. THE System SHALL provide recommendations for minimum departure count when capacity is insufficient

### Requirement 4

**User Story:** As a researcher, I want the environment to correctly track and serve all loaded passengers, so that stranded passenger counts accurately reflect scheduling quality

#### Acceptance Criteria

1. WHEN THE Environment loads Passenger_Data, THE Environment SHALL store all passengers in passenger_arrivals dictionary
2. WHEN current_time matches a passenger arrival_time, THE Environment SHALL add the passenger to the correct station queue
3. WHEN a bus arrives at a station, THE Environment SHALL board waiting passengers up to available_capacity
4. THE Environment SHALL track total_passengers_served for each direction
5. THE Environment SHALL calculate stranded_passengers as total_loaded - total_served at service_end

### Requirement 5

**User Story:** As a researcher, I want to validate the data loading process, so that I can confirm all passengers are correctly loaded before training

#### Acceptance Criteria

1. THE System SHALL provide a validation function that compares loaded passenger count with source data count
2. WHEN validation runs, THE System SHALL report passengers_in_data vs passengers_in_environment for each direction
3. WHEN counts do not match, THE System SHALL identify the discrepancy cause
4. THE System SHALL validate that passenger arrival times are within Service_Window after filtering
5. THE System SHALL validate that all station IDs in loaded passengers exist in Environment
