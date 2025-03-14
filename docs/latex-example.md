# Effects and Mathematical Formulations

This page demonstrates how to use LaTeX in flixOpt documentation and explains the Effects system.

## Effects in flixOpt

Effects in flixOpt represent impacts or metrics related to your energy system, such as costs, emissions, resource consumption, etc. One effect is designated as the optimization objective (typically costs), while others can have constraints.

## Mathematical Formulations

### Storage Model

The state of charge of a storage evolves according to:

$$SOC(t+1) = SOC(t) \cdot (1 - \lambda \cdot \Delta t) + \eta_{charge} \cdot P_{in}(t) \cdot \Delta t - \frac{P_{out}(t)}{\eta_{discharge}} \cdot \Delta t$$

Where:

- $SOC(t)$ is the state of charge at time $t$
- $\lambda$ is the self-discharge rate
- $\eta_{charge}$ is the charging efficiency
- $\eta_{discharge}$ is the discharging efficiency
- $P_{in}(t)$ is the charging power
- $P_{out}(t)$ is the discharging power
- $\Delta t$ is the time step

### Linear Converter Efficiency

For a linear converter, the relationship between input and output is:

$$P_{out}(t) = \eta \cdot P_{in}(t)$$

Where:
- $P_{out}(t)$ is the output power
- $P_{in}(t)$ is the input power
- $\eta$ is the efficiency

### Heat Pump COP

For a heat pump, the relationship is:

$$Q_{th}(t) = COP \cdot P_{el}(t)$$

Where:
- $Q_{th}(t)$ is the heat output
- $P_{el}(t)$ is the electrical input
- $COP$ is the coefficient of performance

### Objective Function

The objective function for cost minimization is:

$$\min \left( \sum_{t=1}^{T} \sum_{c \in C} c_{op}(t) \cdot P_c(t) \cdot \Delta t + \sum_{c \in C} c_{inv} \cdot CAP_c \right)$$

Where:
- $c_{op}(t)$ is the operating cost at time $t$
- $P_c(t)$ is the power of component $c$ at time $t$
- $c_{inv}$ is the investment cost
- $CAP_c$ is the capacity of component $c$

## Effects API Documentation

Effects are created using the `Effect` class:

```python
import flixOpt as fo

# Create a cost effect (optimization objective)
cost_effect = fo.Effect(
    label="costs",
    unit="€",
    description="Total costs",
    is_objective=True  # This effect will be minimized
)

# Create a CO2 emission effect with constraints
co2_effect = fo.Effect(
    label="co2_emissions",
    unit="kg_CO2",
    description="CO2 emissions",
    maximum_total=1000  # Maximum total emissions allowed
)

# Add effects to the system
system.add_effects(cost_effect, co2_effect)
```

## Inline Formulas

You can also use inline formulas like $E = mc^2$ or reference variables like $\eta_{boiler}$ within your text.

## Multiple Equations Example

The efficiency of a CHP unit must satisfy:

$$\eta_{el} + \eta_{th} \leq \eta_{max}$$

The total flow through a bus must be balanced:

$$\sum_{i \in I} F_{i,in}(t) = \sum_{j \in J} F_{j,out}(t)$$

For components with on/off decisions, the flow must satisfy:

$$F_{min} \cdot \delta(t) \leq F(t) \leq F_{max} \cdot \delta(t)$$

Where $\delta(t)$ is a binary variable indicating if the component is on.
