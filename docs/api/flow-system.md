# FlowSystem API Reference

The FlowSystem is the central organizing component in flixOpt, responsible for managing the time series, components, buses, and effects that make up your energy system model.

## FlowSystem Class

::: flixOpt.flow_system.FlowSystem
    options:
      members: true
      show_root_heading: true
      show_source: true

## Examples

### Creating a FlowSystem

```python
import flixOpt as fx
import pandas as pd

# Create the timesteps with hourly steps for one day
timesteps = pd.date_range('2020-01-01', periods=24, freq='h')

# Initialize the FlowSystem with the timesteps
flow_system = fx.FlowSystem(timesteps=timesteps)

# Add components, buses, and effects
heat_bus = fx.Bus("Heat")
flow_system.add_elements(heat_bus)

# Visualize the network
flow_system.plot_network(show=True)
```

### Accessing FlowSystem Components

```python
# Get a list of all components
components = flow_system.components

# Get a specific component by label
if "Boiler" in flow_system.components:
    boiler = flow_system.components["Boiler"]
    
# Get all flows in the flow_system
flows = flow_system.flows

# Get all buses in the flow_system
buses = flow_system.buses
```

### Time Series and Indices

```python
# Get the full time series
full_time = flow_system.time_series

# Get a subset of the time series
indices = range(12)  # First 12 hours
time_subset, time_with_end, dt_hours, total_hours = flow_system.get_time_data_from_indices(indices)
```

### Saving System Information

```python
# Save flow_system information to a JSON file
flow_system.to_json("system_info.json")

# Save flow_system visualization
flow_system.visualize_network(path="system_network.html", show=False)
```
