# FlowSystem

The FlowSystem is the central organizing component in flixOpt, responsible for managing the time series, components, buses, and effects that make up your energy system model.

## API Reference

::: flixOpt.flow_system.FlowSystem
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_object_full_path: false
      show_category_heading: false
      show_source: false
      members_order: source
      heading_level: 3
      docstring_section_style: table

## Usage Examples

```python
import flixOpt as fx
import pandas as pd


# Create timesteps with hourly steps for one day
timesteps = pd.date_range('2023-01-01', steps=24, freq='1h')

# Initialize the FlowSystem
flow_system = fx.FlowSystem(timesteps)

# Add buses, components and effects
heat_bus = fx.Bus("Heat")
electricity_bus = fx.Bus("Electricity")
costs = fx.Effect("costs", "€", "Costs", is_standard=True, is_objective=True)
flow_system.add_elements(heat_bus, electricity_bus, costs)

# You can add components with their connected flows
heat_pump = fx.linear_converters.HeatPump(
  label="HeatPump",
  COP=3.0,
  P_el=fx.Flow("power", electricity_bus.label, effects_per_flow_hour=0.2),
  Q_th=fx.Flow("heat", heat_bus.label)
)
flow_system.add_elements(heat_pump)

# Access components and flow_system structure
print(flow_system.components)  # Dictionary of all components
print(flow_system.buses)  # Dictionary of all buses
print(flow_system.flows)  # Dictionary of all flows

# Visualize the flow_system network
flow_system.plot_network(show=True)

# Save the flow_system definition
flow_system.to_json("flow_system_definition.json")
```
