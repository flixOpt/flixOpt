"""
This module contains the basic components of the flixOpt framework.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple, Union

import linopy
import numpy as np
import pandas as pd

from . import utils
from .core import NumericData, NumericDataTS, Scalar, TimeSeries, TimeSeriesCollection
from .elements import Component, ComponentModel, Flow
from .features import InvestmentModel, MultipleSegmentsModel, OnOffModel
from .interface import InvestParameters, OnOffParameters
from .structure import SystemModel, register_class_for_io

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixOpt')

@register_class_for_io
class LinearConverter(Component):
    """
    Converts one FLow into another via linear conversion factors
    """

    def __init__(
        self,
        label: str,
        inputs: List[Flow],
        outputs: List[Flow],
        on_off_parameters: OnOffParameters = None,
        conversion_factors: List[Dict[str, NumericDataTS]] = None,
        segmented_conversion_factors: Dict[str, List[Tuple[NumericDataTS, NumericDataTS]]] = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        inputs : input flows.
        outputs : output flows.
        on_off_parameters: Information about on and off states. See class OnOffParameters.
        conversion_factors : linear relation between flows.
            Either 'conversion_factors' or 'segmented_conversion_factors' can be used!
            example heat pump:
        segmented_conversion_factors :  Segmented linear relation between flows.
            Each Flow gets a List of Segments assigned.
            If FLows need to be 0 (or Off), include a "Zero-Segment" "(0, 0)", or use on_off_parameters
            Either 'segmented_conversion_factors' or 'conversion_factors' can be used!
            --> "gaps" can be expressed by a segment not starting at the end of the prior segment : [(1,3), (4,5)]
            --> "points" can expressed as segment with same begin and end : [(3,3), (4,4)]

        """
        super().__init__(label, inputs, outputs, on_off_parameters, meta_data=meta_data)
        self.conversion_factors = conversion_factors or []
        self.segmented_conversion_factors = segmented_conversion_factors or {}

    def create_model(self, model: SystemModel) -> 'LinearConverterModel':
        self._plausibility_checks()
        self.model = LinearConverterModel(model, self)
        return self.model

    def _plausibility_checks(self) -> None:
        if not self.conversion_factors and not self.segmented_conversion_factors:
            raise Exception('Either conversion_factors or segmented_conversion_factors must be defined!')
        if self.conversion_factors and self.segmented_conversion_factors:
            raise Exception('Only one of conversion_factors or segmented_conversion_factors can be defined, not both!')

        if self.conversion_factors:
            if self.degrees_of_freedom <= 0:
                raise Exception(
                    f'Too Many conversion_factors_specified. Care that you use less conversion_factors '
                    f'then inputs + outputs!! With {len(self.inputs + self.outputs)} inputs and outputs, '
                    f'use not more than {len(self.inputs + self.outputs) - 1} conversion_factors!'
                )

            for conversion_factor in self.conversion_factors:
                for flow in conversion_factor:
                    if flow not in self.flows:
                        raise Exception(
                            f'{self.label}: Flow {flow} in conversion_factors is not in inputs/outputs'
                        )
        if self.segmented_conversion_factors:
            for flow in self.flows.values():
                if isinstance(flow.size, InvestParameters) and flow.size.fixed_size is None:
                    raise Exception(
                        f'segmented_conversion_factors (in {self.label_full}) and variable size '
                        f'(in flow {flow.label_full}) do not make sense together!'
                    )

    def transform_data(self, flow_system: 'FlowSystem'):
        super().transform_data(flow_system)
        if self.conversion_factors:
            self.conversion_factors = self._transform_conversion_factors(flow_system)
        else:
            segmented_conversion_factors = {}
            for flow, segments in self.segmented_conversion_factors.items():
                segmented_conversion_factors[flow] = [
                    (
                        flow_system.create_time_series(f'{self.flows[flow].label_full}|Stützstelle|{idx}a', segment[0]),
                        flow_system.create_time_series(f'{self.flows[flow].label_full}|Stützstelle|{idx}b', segment[1]),
                    )
                    for idx, segment in enumerate(segments)
                ]
            self.segmented_conversion_factors = segmented_conversion_factors

    def _transform_conversion_factors(self, flow_system: 'FlowSystem') -> List[Dict[str, TimeSeries]]:
        """macht alle Faktoren, die nicht TimeSeries sind, zu TimeSeries"""
        list_of_conversion_factors = []
        for idx, conversion_factor in enumerate(self.conversion_factors):
            transformed_dict = {}
            for flow, values in conversion_factor.items():
                # TODO: Might be better to use the label of the component instead of the flow
                transformed_dict[flow] = flow_system.create_time_series(
                    f'{self.flows[flow].label_full}|conversion_factor{idx}', values
                )
            list_of_conversion_factors.append(transformed_dict)
        return list_of_conversion_factors

    @property
    def degrees_of_freedom(self):
        return len(self.inputs + self.outputs) - len(self.conversion_factors)


@register_class_for_io
class Storage(Component):
    """
    Klasse Storage
    """

    # TODO: Dabei fällt mir auf. Vielleicht sollte man mal überlegen, ob man für Ladeleistungen bereits in dem
    #  jeweiligen Zeitschritt mit einem Verlust berücksichtigt. Zumindest für große Zeitschritte bzw. große Verluste
    #  eventuell relevant.
    #  -> Sprich: speicherverlust = charge_state(t) * relative_loss_per_hour * dt + 0.5 * Q_lade(t) * dt * relative_loss_per_hour * dt
    #  -> müsste man aber auch für den sich ändernden Ladezustand berücksichtigten

    def __init__(
        self,
        label: str,
        charging: Flow,
        discharging: Flow,
        capacity_in_flow_hours: Union[Scalar, InvestParameters],
        relative_minimum_charge_state: NumericData = 0,
        relative_maximum_charge_state: NumericData = 1,
        initial_charge_state: Optional[Union[Scalar, Literal['lastValueOfSim']]] = 0,
        minimal_final_charge_state: Optional[Scalar] = None,
        maximal_final_charge_state: Optional[Scalar] = None,
        eta_charge: NumericData = 1,
        eta_discharge: NumericData = 1,
        relative_loss_per_hour: NumericData = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        constructor of storage

        Parameters
        ----------
        label : str
            description.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        charging : Flow
            ingoing flow.
        discharging : Flow
            outgoing flow.
        capacity_in_flow_hours : Scalar or InvestParameter
            nominal capacity of the storage
        relative_minimum_charge_state : float or TS, optional
            minimum relative charge state. The default is 0.
        relative_maximum_charge_state : float or TS, optional
            maximum relative charge state. The default is 1.
        initial_charge_state : None, float (0...1), 'lastValueOfSim',  optional
            storage charge_state at the beginning. The default is 0.
            float: defined charge_state at start of first timestep
            None: free to choose by optimizer
            'lastValueOfSim': chargeState0 is equal to chargestate of last timestep ("closed simulation")
        minimal_final_charge_state : float or None, optional
            minimal value of chargeState at the end of timeseries.
        maximal_final_charge_state : float or None, optional
            maximal value of chargeState at the end of timeseries.
        eta_charge : float, optional
            efficiency factor of charging/loading. The default is 1.
        eta_discharge : TYPE, optional
            efficiency factor of uncharging/unloading. The default is 1.
        relative_loss_per_hour : float or TS. optional
            loss per chargeState-Unit per hour. The default is 0.
        prevent_simultaneous_charge_and_discharge : boolean, optional
            should simultaneously Loading and Unloading be avoided? (Attention, Performance maybe becomes worse with avoidInAndOutAtOnce=True). The default is True.
        """
        # TODO: fixed_relative_chargeState implementieren
        super().__init__(
            label,
            inputs=[charging],
            outputs=[discharging],
            prevent_simultaneous_flows=[charging, discharging] if prevent_simultaneous_charge_and_discharge else None,
            meta_data=meta_data,
        )

        self.charging = charging
        self.discharging = discharging
        self.capacity_in_flow_hours = capacity_in_flow_hours
        self.relative_minimum_charge_state: NumericDataTS = relative_minimum_charge_state
        self.relative_maximum_charge_state: NumericDataTS = relative_maximum_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge: NumericDataTS = eta_charge
        self.eta_discharge: NumericDataTS = eta_discharge
        self.relative_loss_per_hour: NumericDataTS = relative_loss_per_hour
        self.prevent_simultaneous_charge_and_discharge = prevent_simultaneous_charge_and_discharge

    def create_model(self, model: SystemModel) -> 'StorageModel':
        self.model = StorageModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        super().transform_data(flow_system)
        self.relative_minimum_charge_state = flow_system.create_time_series(
            f'{self.label_full}|relative_minimum_charge_state', self.relative_minimum_charge_state, needs_extra_timestep=True
        )
        self.relative_maximum_charge_state = flow_system.create_time_series(
            f'{self.label_full}|relative_maximum_charge_state', self.relative_maximum_charge_state, needs_extra_timestep=True
        )
        self.eta_charge = flow_system.create_time_series(f'{self.label_full}|eta_charge', self.eta_charge)
        self.eta_discharge = flow_system.create_time_series(f'{self.label_full}|eta_discharge', self.eta_discharge)
        self.relative_loss_per_hour = flow_system.create_time_series(
            f'{self.label_full}|relative_loss_per_hour', self.relative_loss_per_hour
        )
        if isinstance(self.capacity_in_flow_hours, InvestParameters):
            self.capacity_in_flow_hours.transform_data(flow_system)


@register_class_for_io
class Transmission(Component):
    # TODO: automatic on-Value in Flows if loss_abs
    # TODO: loss_abs must be: investment_size * loss_abs_rel!!!
    # TODO: investmentsize only on 1 flow
    # TODO: automatic investArgs for both in-flows (or alternatively both out-flows!)
    # TODO: optional: capacities should be recognised for losses

    def __init__(
        self,
        label: str,
        in1: Flow,
        out1: Flow,
        in2: Optional[Flow] = None,
        out2: Optional[Flow] = None,
        relative_losses: Optional[NumericDataTS] = None,
        absolute_losses: Optional[NumericDataTS] = None,
        on_off_parameters: OnOffParameters = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
    ):
        """
        Initializes a Transmission component (Pipe, cable, ...) that models the flows between two sides
        with potential losses.

        Parameters
        ----------
        label : str
            The name of the transmission component.
        in1 : Flow
            The inflow at side A. Pass InvestmentParameters here.
        out1 : Flow
            The outflow at side B.
        in2 : Optional[Flow], optional
            The optional inflow at side B.
            If in1 got Investmentparameters, the size of this Flow will be equal to in1 (with no extra effects!)
        out2 : Optional[Flow], optional
            The optional outflow at side A.
        relative_losses : Optional[NumericDataTS], optional
            The relative loss between inflow and outflow, e.g., 0.02 for 2% loss.
        absolute_losses : Optional[NumericDataTS], optional
            The absolute loss, occur only when the Flow is on. Induces the creation of the ON-Variable
        on_off_parameters : OnOffParameters, optional
            Parameters defining the on/off behavior of the component.
        prevent_simultaneous_flows_in_both_directions : bool, default=True
            If True, prevents simultaneous flows in both directions.
        """
        super().__init__(
            label,
            inputs=[flow for flow in (in1, in2) if flow is not None],
            outputs=[flow for flow in (out1, out2) if flow is not None],
            on_off_parameters=on_off_parameters,
            prevent_simultaneous_flows=None
            if in2 is None or prevent_simultaneous_flows_in_both_directions is False
            else [in1, in2],
        )
        self.in1 = in1
        self.out1 = out1
        self.in2 = in2
        self.out2 = out2

        self.relative_losses = relative_losses
        self.absolute_losses = absolute_losses

    def _plausibility_checks(self):
        # check buses:
        if self.in2 is not None:
            assert self.in2.bus == self.out1.bus, (
                f'Output 1 and Input 2 do not start/end at the same Bus: {self.out1.bus=}, {self.in2.bus=}'
            )
        if self.out2 is not None:
            assert self.out2.bus == self.in1.bus, (
                f'Input 1 and Output 2 do not start/end at the same Bus: {self.in1.bus=}, {self.out2.bus=}'
            )
        # Check Investments
        for flow in [self.out1, self.in2, self.out2]:
            if flow is not None and isinstance(flow.size, InvestParameters):
                raise ValueError(
                    'Transmission currently does not support separate InvestParameters for Flows. '
                    'Please use Flow in1. The size of in2 is equal to in1. THis is handled internally'
                )

    def create_model(self, model) -> 'TransmissionModel':
        self.model = TransmissionModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        super().transform_data(flow_system)
        self.relative_losses = flow_system.create_time_series(
            f'{self.label_full}|relative_losses', self.relative_losses
        )
        self.absolute_losses = flow_system.create_time_series(
            f'{self.label_full}|absolute_losses', self.absolute_losses
        )


class TransmissionModel(ComponentModel):
    def __init__(self, model: SystemModel, element: Transmission):
        super().__init__(model, element)
        self.element: Transmission = element
        self.on_off: Optional[OnOffModel] = None

    def do_modeling(self):
        """Initiates all FlowModels"""
        # Force On Variable if absolute losses are present
        if (self.element.absolute_losses is not None) and np.any(self.element.absolute_losses.active_data != 0):
            for flow in self.element.inputs + self.element.outputs:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()

        # Make sure either None or both in Flows have InvestParameters
        if self.element.in2 is not None:
            if isinstance(self.element.in1.size, InvestParameters) and not isinstance(
                self.element.in2.size, InvestParameters
            ):
                self.element.in2.size = InvestParameters(maximum_size=self.element.in1.size.maximum_size)

        super().do_modeling()

        # first direction
        self.create_transmission_equation('dir1', self.element.in1, self.element.out1)

        # second direction:
        if self.element.in2 is not None:
            self.create_transmission_equation('dir2', self.element.in2, self.element.out2)

        # equate size of both directions
        if isinstance(self.element.in1.size, InvestParameters) and self.element.in2 is not None:
            # eq: in1.size = in2.size
            self.add(self._model.add_constraints(
                self.element.in1.model._investment.size == self.element.in2.model._investment.size,
                name=f'{self.label_full}|same_size'),
                'same_size'
            )

    def create_transmission_equation(self, name: str, in_flow: Flow, out_flow: Flow) -> linopy.Constraint:
        """Creates an Equation for the Transmission efficiency and adds it to the model"""
        # eq: out(t) + on(t)*loss_abs(t) = in(t)*(1 - loss_rel(t))
        con_transmission = self.add(self._model.add_constraints(
            out_flow.model.flow_rate == -in_flow.model.flow_rate * (self.element.relative_losses.active_data - 1),
            name=f'{self.label_full}|{name}'),
            name
        )

        if self.element.absolute_losses is not None:
            con_transmission.lhs += in_flow.model.on_off.on * self.element.absolute_losses.active_data

        return con_transmission


class LinearConverterModel(ComponentModel):
    def __init__(self, model: SystemModel, element: LinearConverter):
        super().__init__(model, element)
        self.element: LinearConverter = element
        self.on_off: Optional[OnOffModel] = None

    def do_modeling(self):
        super().do_modeling()

        # conversion_factors:
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs)
            all_output_flows = set(self.element.outputs)

            # für alle linearen Gleichungen:
            for i, conv_factors in enumerate(self.element.conversion_factors):
                used_flows = set([self.element.flows[flow_label] for flow_label in conv_factors])
                used_inputs: Set = all_input_flows & used_flows
                used_outputs: Set = all_output_flows & used_flows

                self.add(
                    self._model.add_constraints(
                        sum([flow.model.flow_rate * conv_factors[flow.label].active_data for flow in used_inputs])
                        ==
                        sum([flow.model.flow_rate * conv_factors[flow.label].active_data for flow in used_outputs]),
                        name=f'{self.label_full}|conversion_{i}'
                    )
                )

        # (linear) segments:
        else:
            # TODO: Improve Inclusion of OnOffParameters. Instead of creating a Binary in every flow, the binary could only be part of the Segment itself
            segments: Dict[str, List[Tuple[NumericData, NumericData]]] = {
                self.element.flows[flow].model.flow_rate.name: [
                    (ts1.active_data, ts2.active_data) for ts1, ts2 in self.element.segmented_conversion_factors[flow]
                ]
                for flow in self.element.flows
            }
            linear_segments = MultipleSegmentsModel(
                self._model, self.label_of_element, segments, self.on_off.on if self.on_off is not None else None
            )  # TODO: Add Outside_segments Variable (On)
            linear_segments.do_modeling()
            self.sub_models.append(linear_segments)


class StorageModel(ComponentModel):
    """Model of Storage"""

    def __init__(self, model: SystemModel, element: Storage):
        super().__init__(model, element)
        self.element: Storage = element
        self.charge_state: Optional[linopy.Variable] = None
        self.netto_discharge: Optional[linopy.Variable] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self):
        super().do_modeling()

        lb, ub = self.absolute_charge_state_bounds
        self.charge_state = self.add(self._model.add_variables(
            lower=lb, upper=ub, coords=self._model.coords_extra,
            name=f'{self.label_full}|charge_state'),
            'charge_state'
        )
        self.netto_discharge = self.add(self._model.add_variables(
            coords=self._model.coords, name=f'{self.label_full}|netto_discharge'),
            'netto_discharge'
        )
        # netto_discharge:
        # eq: nettoFlow(t) - discharging(t) + charging(t) = 0
        self.add(self._model.add_constraints(
            self.netto_discharge == self.element.discharging.model.flow_rate - self.element.charging.model.flow_rate,
            name=f'{self.label_full}|netto_discharge'),
            'netto_discharge'
        )

        charge_state = self.charge_state
        rel_loss = self.element.relative_loss_per_hour.active_data
        hours_per_step = self._model.hours_per_step
        charge_rate = self.element.charging.model.flow_rate
        discharge_rate = self.element.discharging.model.flow_rate
        eff_charge = self.element.eta_charge.active_data
        eff_discharge = self.element.eta_discharge.active_data

        self.add(self._model.add_constraints(
            charge_state.isel(time=slice(1, None))
            ==
            charge_state.isel(time=slice(None, -1)) * (1 - rel_loss * hours_per_step)
            + charge_rate * eff_charge * hours_per_step
            - discharge_rate * eff_discharge * hours_per_step,
            name=f'{self.label_full}|charge_state'),
            'charge_state'
        )

        if isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            self._investment = InvestmentModel(
                model=self._model,
                label_of_element=self.label_of_element,
                parameters=self.element.capacity_in_flow_hours,
                defining_variable=self.charge_state,
                relative_bounds_of_defining_variable=self.relative_charge_state_bounds,
            )
            self.sub_models.append(self._investment)
            self._investment.do_modeling()

        # Initial charge state
        self._initial_and_final_charge_state()

    def _initial_and_final_charge_state(self):
        if self.element.initial_charge_state is not None:
            name_short = 'initial_charge_state'
            name = f'{self.label_full}|{name_short}'

            if utils.is_number(self.element.initial_charge_state):
                self.add(self._model.add_constraints(
                    self.charge_state.isel(time=0) == self.element.initial_charge_state,
                    name=name),
                    name_short
                )
            elif self.element.initial_charge_state == 'lastValueOfSim':
                self.add(self._model.add_constraints(
                    self.charge_state.isel(time=0) == self.charge_state.isel(time=-1),
                    name=name),
                    name_short
                )
            else:  # TODO: Validation in Storage Class, not in Model
                raise Exception(f'initial_charge_state has undefined value: {self.element.initial_charge_state}')

        if self.element.maximal_final_charge_state is not None:
            self.add(self._model.add_constraints(
                self.charge_state.isel(time=-1) <= self.element.maximal_final_charge_state,
                name=f'{self.label_full}|final_charge_max'),
                'final_charge_max'
            )

        if self.element.minimal_final_charge_state is not None:
            self.add(self._model.add_constraints(
                self.charge_state.isel(time=-1) >= self.element.minimal_final_charge_state,
                name=f'{self.label_full}|final_charge_min'),
                'final_charge_min'
            )

    @property
    def absolute_charge_state_bounds(self) -> Tuple[NumericData, NumericData]:
        relative_lower_bound, relative_upper_bound = self.relative_charge_state_bounds
        if not isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            return (
                relative_lower_bound * self.element.capacity_in_flow_hours,
                relative_upper_bound * self.element.capacity_in_flow_hours,
            )
        else:
            return (
                relative_lower_bound * self.element.capacity_in_flow_hours.minimum_size,
                relative_upper_bound * self.element.capacity_in_flow_hours.maximum_size,
            )

    @property
    def relative_charge_state_bounds(self) -> Tuple[NumericData, NumericData]:
        return (
            self.element.relative_minimum_charge_state.active_data,
            self.element.relative_maximum_charge_state.active_data,
        )


@register_class_for_io
class SourceAndSink(Component):
    """
    class for source (output-flow) and sink (input-flow) in one commponent
    """

    # source : Flow
    # sink   : Flow

    def __init__(
        self,
        label: str,
        source: Flow,
        sink: Flow,
        prevent_simultaneous_sink_and_source: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name of sourceAndSink
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        source : Flow
            output-flow of this component
        sink : Flow
            input-flow of this component
        prevent_simultaneous_sink_and_source: boolean. Default ist True.
            True: inflow and outflow are not allowed to be both non-zero at same timestep.
            False: inflow and outflow are working independently.

        """
        super().__init__(
            label,
            inputs=[sink],
            outputs=[source],
            prevent_simultaneous_flows=[sink, source] if prevent_simultaneous_sink_and_source is True else None,
            meta_data=meta_data,
        )
        self.source = source
        self.sink = sink
        self.prevent_simultaneous_sink_and_source = prevent_simultaneous_sink_and_source


@register_class_for_io
class Source(Component):
    def __init__(self, label: str, source: Flow, meta_data: Optional[Dict] = None):
        """
        Parameters
        ----------
        label : str
            name of source
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        source : Flow
            output-flow of source
        """
        super().__init__(label, outputs=[source], meta_data=meta_data)
        self.source = source


@register_class_for_io
class Sink(Component):
    def __init__(self, label: str, sink: Flow, meta_data: Optional[Dict] = None):
        """
        constructor of sink

        Parameters
        ----------
        label : str
            name of sink.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        sink : Flow
            input-flow of sink
        """
        super().__init__(label, inputs=[sink], meta_data=meta_data)
        self.sink = sink
