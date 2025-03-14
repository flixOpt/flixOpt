# Calculation API Reference

The calculation module contains classes for solving optimization problems in different ways. flixOpt offers three main calculation modes, each with different performance characteristics and use cases.

## Calculation Base Class

::: flixOpt.calculation.Calculation
    options:
      members: true
      show_root_heading: true
      show_source: true

## Full Calculation

The `FullCalculation` class solves the entire optimization problem at once:

::: flixOpt.calculation.FullCalculation
    options:
      members: true
      show_root_heading: true
      show_source: true

## Segmented Calculation

The `SegmentedCalculation` class splits the problem into segments to improve performance:

::: flixOpt.calculation.SegmentedCalculation
    options:
      members: true
      show_root_heading: true
      show_source: true

## Aggregated Calculation

The `AggregatedCalculation` class uses typical periods to reduce computational requirements:

::: flixOpt.calculation.AggregatedCalculation
    options:
      members: true
      show_root_heading: true
      show_source: true

## Aggregation Parameters

::: flixOpt.aggregation.AggregationParameters
    options:
      members: true
      show_root_heading: true
      show_source: true

## Examples

### Full Calculation Example

```python
import flixOpt as fo

# Create system and add components
system = fo.FlowSystem(time_series)
# ... add components, buses, etc.

# Create a full calculation
calculation = fo.FullCalculation("Example", system)

# Choose a solver
solver = fo.HighsSolver()

# Run the calculation
calculation.do_modeling()
calculation.solve(solver, save_results=True)

# Access results
results = calculation.results()
```

### Segmented Calculation Example

```python
import flixOpt as fo

# Create system and add components
system = fo.FlowSystem(time_series)
# ... add components, buses, etc.

# Create a segmented calculation
segment_length = 24  # 24 time steps per segment
overlap_length = 6   # 6 time steps overlap between segments
calculation = fo.SegmentedCalculation(
    "Segmented_Example", 
    system,
    segment_length=segment_length,
    overlap_length=overlap_length
)

# Choose a solver
solver = fo.HighsSolver()

# Run the calculation
calculation.do_modeling_and_solve(solver, save_results=True)

# Access results - combining arrays from all segments
results = calculation.results(combined_arrays=True)
```

### Aggregated Calculation Example

```python
import flixOpt as fo

# Create system and add components
system = fo.FlowSystem(time_series)
# ... add components, buses, etc.

# Define aggregation parameters
aggregation_params = fo.AggregationParameters(
    hours_per_period=24,        # 24 hours per typical period
    nr_of_periods=10,           # 10 typical periods
    fix_storage_flows=False,    # Don't fix storage flows
    aggregate_data_and_fix_non_binary_vars=True  # Aggregate all time series data
)

# Create an aggregated calculation
calculation = fo.A