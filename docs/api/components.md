# Components API Reference

Components in flixOpt represent physical entities that consume, produce, or transform energy and material flows. This page documents the core component classes available in flixOpt.

## Base Component

The `Component` class is the base class for all components in flixOpt:

::: flixOpt.elements.Component
    options:
      members: true
      show_root_heading: true
      show_source: true

## Storage

The `Storage` class represents energy or material storage components:

::: flixOpt.components.Storage
    options:
      members: true
      show_root_heading: true
      show_source: true

### Storage Model

::: flixOpt.components.StorageModel
    options:
      members: true
      show_root_heading: true
      show_source: true

## LinearConverter

The `LinearConverter` class handles linear conversion between flows:

::: flixOpt.components.LinearConverter
    options:
      members: true
      show_root_heading: true
      show_source: true

### LinearConverter Model

::: flixOpt.components.LinearConverterModel
    options:
      members: true
      show_root_heading: true
      show_source: true

## Transmission

The `Transmission` class models the flows between two sides with potential losses:

::: flixOpt.components.Transmission
    options:
      members: true
      show_root_heading: true
      show_source: true

### Transmission Model

::: flixOpt.components.TransmissionModel
    options:
      members: true
      show_root_heading: true
      show_source: true

## Source, Sink, and SourceAndSink

Classes for sources and sinks in the system:

::: flixOpt.components.Source
    options:
      members: true
      show_root_heading: true
      show_source: true

::: flixOpt.components.Sink
    options:
      members: true
      show_root_heading: true
      show_source: true

::: flixOpt.components.SourceAndSink
    options:
      members: true
      show_root_heading: true
      show_source: true

## Examples

### Creating a LinearConverter

```python
import flixOpt as fo

# Create buses
electricity_bus = fo.Bus("Electricity")
heat_bus = fo.Bus("Heat")

# Create flows
power_input = fo.Flow("power_in", electricity_bus)
heat_output = fo.Flow("heat_out", heat_bus)

# Create a heat pump with COP = 3
heat_pump = fo.components.LinearConverter(
    label="HeatPump",
    inputs=[power_input],
    outputs=[heat_output],
    conversion_factors=[{power_input: 3, heat_output: 1}]
)
```

### Creating a Storage

```python
import flixOpt as fo

# Create a bus
heat_bus = fo.Bus("Heat")

# Create charging and discharging flows
charging = fo.Flow("charging", heat_bus)
discharging = fo.Flow("discharging", heat_bus)

# Create a thermal storage
thermal_storage = fo.components.Storage(
    label="ThermalStorage",
    charging=charging,
    discharging=discharging,
    capacity_in_flow_hours=1000,  # 1000 kWh capacity
    relative_loss_per_hour=0.01,  # 1% loss per hour
    eta_charge=0.95,              # 95% charging efficiency
    eta_discharge=0.95            # 95% discharging efficiency
)
```

### Creating a Transmission Component

```python
import flixOpt as fo

# Create buses
bus_a = fo.Bus("Location_A")
bus_b = fo.Bus("Location_B")

# Create flows
flow_a_to_b = fo.Flow("flow_a_to_b", bus_a)
flow_b_to_a = fo.Flow("flow_b_to_a", bus_b)

# Create a transmission component with 5% losses
transmission = fo.components.Transmission(
    label="Transmission_Line",
    in1=flow_a_to_b,
    out1=flow_b_to_a,
    relative_losses=0.05
)
```
