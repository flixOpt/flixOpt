# flixOpt: Energy and Material Flow Optimization Framework

**flixOpt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP). Combining flexibility and efficiency, it provides a powerful platform for both dispatch and investment optimization challenges.

## 🚀 Introduction

flixOpt was developed by [TU Dresden](https://github.com/gewv-tu-dresden) as part of the SMARTBIOGRID project, funded by the German Federal Ministry for Economic Affairs and Energy. Building on the Matlab-based flixOptMat framework, flixOpt also incorporates concepts from [oemof/solph](https://github.com/oemof/oemof-solph).

Although flixOpt is in its early stages, it is fully functional and ready for experimentation. Feedback and collaboration are highly encouraged to help shape its future.

## 🌟 Key Features

- **High-level Interface** with low-level control
  - User-friendly interface for defining energy systems
  - Fine-grained control for advanced configurations
  - Pre-defined components like CHP, Heat Pump, Cooling Tower, etc.

- **Investment Optimization**
  - Combined dispatch and investment optimization
  - Size and discrete investment decisions
  - Integration with On/Off variables and constraints

- **Multiple Effects**
  - Couple effects (e.g., specific CO2 costs)
  - Set constraints (e.g., max CO2 emissions)
  - Easily switch optimization targets (e.g., costs vs CO2)

- **Calculation Modes**
  - **Full Mode** - Exact solutions with high computational requirements
  - **Segmented Mode** - Speed up complex systems with variable time overlap
  - **Aggregated Mode** - Typical periods for large-scale simulations

## 📦 Installation

Install flixOpt directly using pip:

```bash
pip install git+https://github.com/flixOpt/flixOpt.git
```

For full functionality including visualization and time series aggregation:

```bash
pip install "flixOpt[full] @ git+https://github.com/flixOpt/flixOpt.git"
```

## 🖥️ Quick Example

```python
import flixOpt as fo
import numpy as np

# Create timesteps
time_series = fo.create_datetime_array('2023-01-01', steps=24, freq='1h')
system = fo.FlowSystem(time_series)

# Create buses
heat_bus = fo.Bus("Heat")
electricity_bus = fo.Bus("Electricity")

# Create flows
heat_demand = fo.Flow(
    label="heat_demand",
    bus=heat_bus,
    fixed_relative_profile=100*np.sin(np.linspace(0, 2*np.pi, 24))**2 + 50
)

# Create a heat pump component
heat_pump = fo.linear_converters.HeatPump(
    label="HeatPump",
    COP=3.0,
    P_el=fo.Flow("power", electricity_bus),
    Q_th=fo.Flow("heat", heat_bus)
)

# Add everything to the system
system.add_elements(heat_bus, electricity_bus)
system.add_components(heat_pump)
```

## ⚙️ How It Works

flixOpt transforms your energy system model into a mathematical optimization problem, solves it using state-of-the-art solvers, and returns the optimal operation strategy and investment decisions.

## 📚 Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Concepts](concepts/overview.md) - Core concepts and architecture
- [Examples](examples/basic.md) - Usage examples
- [API Reference](api/flow-system.md) - Full API documentation

## 🛠️ Compatible Solvers

flixOpt works with various solvers:

- HiGHS (installed by default)
- CBC
- GLPK
- Gurobi
- CPLEX

## 📝 Citation

If you use flixOpt in your research or project, please cite:

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)
