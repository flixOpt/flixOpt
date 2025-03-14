::: flixOpt.components.Storage

### Creating a Storage

```python
import flixOpt as fx

thermal_storage = fx.Storage(
    label="ThermalStorage",
    charging=fx.Flow("charging", "Wärme", size=100),
    discharging=fx.Flow("discharging", "Wärme", size=100),
    capacity_in_flow_hours=1000,  # 1000 kWh capacity
    relative_loss_per_hour=0.01,  # 1% loss per hour
    eta_charge=0.95,              # 95% charging efficiency
    eta_discharge=0.95            # 95% discharging efficiency
)
```
