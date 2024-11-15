# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:05:50 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
import json
import pathlib
from typing import Dict, List, Tuple, Literal, Optional, Union
import datetime

import yaml
import numpy as np
import pandas as pd
import plotly

from flixOpt import utils
from flixOpt import plotting

logger = logging.getLogger('flixOpt')


class ElementResults:
    def __init__(self, infos: Dict, data: Dict):
        self.all_infos = infos
        self.all_results = data
        self.label = self.all_infos['label']

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label})'


class CalculationResults:
    def __init__(self, calculation_name: str, folder: str) -> None:
        self._path_infos = (pathlib.Path(folder) / f'{calculation_name}_info.yaml').resolve().as_posix()
        self._path_results = (pathlib.Path(folder) / f'{calculation_name}_data.json').resolve().as_posix()

        with open(self._path_infos, 'rb') as f:
            self.all_infos: Dict = yaml.safe_load(f)

        with open(self._path_results, 'rb') as f:
            self.all_results: Dict = json.load(f)
        self.all_results = utils.convert_numeric_lists_to_arrays(self.all_results)

        self.component_results: Dict[str, ComponentResults] = {}
        self.effect_results: Dict[str, EffectResults] = {}
        self.bus_results: Dict[str, BusResults] = {}

        self.time_with_end = np.array([datetime.datetime.fromisoformat(date) for date in self.all_results['Time']]).astype('datetime64')
        self.time = self.time_with_end[:-1]
        self.time_intervals_in_hours = np.array(self.all_results['Time intervals in hours'])

        self._construct_component_results()
        self._construct_bus_results()
        self._construct_effect_results()

    def _construct_component_results(self):
        comp_results = self.all_results['Components']
        comp_infos = self.all_infos['FlowSystem']['Components']
        assert comp_results.keys() == comp_infos.keys(), \
            f'Missing Component or mismatched keys: {comp_results.keys() ^ comp_infos.keys()}'

        for key in comp_results.keys():
            infos, results = comp_infos[key], comp_results[key]
            res = ComponentResults(infos, results)
            self.component_results[res.label] = res

    def _construct_effect_results(self):
        effect_results = self.all_results['Effects']
        effect_infos = self.all_infos['FlowSystem']['Effects']
        effect_infos['penalty'] = {'label': 'Penalty'}
        assert effect_results.keys() == effect_infos.keys(), \
            f'Missing Effect or mismatched keys: {effect_results.keys() ^ effect_infos.keys()}'

        for key in effect_results.keys():
            infos, results = effect_infos[key], effect_results[key]
            res = EffectResults(infos, results)
            self.effect_results[res.label] = res

    def _construct_bus_results(self):
        """ This has to be called after _construct_component_results(), as its using the Flows from the Components"""
        bus_results = self.all_results['Buses']
        bus_infos = self.all_infos['FlowSystem']['Buses']
        assert bus_results.keys() == bus_infos.keys(), \
            f'Missing Bus or mismatched keys: {bus_results.keys() ^ bus_infos.keys()}'

        for bus_label in bus_results.keys():
            infos, results = bus_infos[bus_label], bus_results[bus_label]
            inputs = [flow for flow in self.flow_results().values() if bus_label==flow.bus_label and not flow.is_input_in_component]
            outputs = [flow for flow in self.flow_results().values() if bus_label==flow.bus_label and flow.is_input_in_component]
            res = BusResults(infos, results, inputs, outputs)
            self.bus_results[res.label] = res

    def flow_results(self) -> Dict[str, 'FlowResults']:
        return {flow.label_full: flow
                for comp in self.component_results.values()
                for flow in comp.inputs + comp.outputs}

    def to_dataframe(self,
                     label: str,
                     variable_name: str = 'flow_rate',
                     input_factor: Optional[Literal[1, -1]] = -1,
                     output_factor: Optional[Literal[1, -1]] = 1,
                     threshold: Optional[float] = 1e-5,
                     with_last_time_step: bool = True,
                     ) -> pd.DataFrame:
        """
        Gets results from an Element with the specified label for the specified Variable.
        The Element can either be a Component or a Bus or a  Flow (full label).
        Returns a Dataframe with a Datetime index, typically including the last step.
        """
        comp_or_bus = {**self.component_results, **self.bus_results}.get(label, None)
        if comp_or_bus is not None:
            df = comp_or_bus.to_dataframe(variable_name, input_factor, output_factor,)
        else:
            flow = self.flow_results().get(label, None)
            df = flow.to_dataframe(variable_name)
        if threshold is not None:
            df = df.loc[:, ((df > threshold) | (df < -1*threshold)).any()]  # Check if any value exceeds the threshold

        if with_last_time_step:
            if len(df) == len(self.time):
                df.loc[len(df)] = df.iloc[-1]
            df.index = self.time_with_end
        elif len(df) == len(self.time_with_end):
            df.index = self.time_with_end
        else:
            df.index = self.time

        return df

    def plot_operation(self,
                       label: str,
                       mode: Literal['bar', 'line', 'area', 'heatmap'] = 'area',
                       variable_name: str = 'flow_rate',
                       heatmap_periods: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
                       heatmap_steps_per_period: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
                       engine: Literal['plotly', 'matplotlib'] = 'plotly',
                       show: bool = True):
        data = self.to_dataframe(label, variable_name)
        title = f'{variable_name.replace("_", " ").title()} of {label}'
        if engine == 'plotly':
            if mode == 'heatmap':
                if not np.all(self.time_intervals_in_hours == self.time_intervals_in_hours[0]):
                    logger.warning(
                        'Heat map plotting with irregular time intervals in time series can lead to unwanted effects')
                heatmap_data = plotting.heat_map_data_from_df(data, heatmap_periods, heatmap_steps_per_period, 'ffill')
                return plotting.heat_map_plotly(heatmap_data, show=show, title=title)
            else:
                return plotting.with_plotly(data=data, mode=mode, show=show, title=title)

        elif engine == 'matplotlib':
            if mode == 'heatmap':
                heatmap_data = plotting.heat_map_data_from_df(data, heatmap_periods, heatmap_steps_per_period, 'ffill')
                return plotting.heat_map_matplotlib(heatmap_data, show=show)
            else:
                return plotting.with_matplotlib(data=data, mode=mode, show=show)
        else:
            raise ValueError(f'Unknown Engine: {engine=}')

    def plot_storage(self,
                     label: str,
                     variable_name: str = 'flow_rate',
                     mode: Literal['bar', 'line', 'area'] = 'area',
                     show: bool = True):
        fig = self.plot_operation(label, mode, variable_name, engine='plotly', show=False)
        fig.add_trace(plotly.graph_objs.Scatter(
            x=self.time_with_end,
            y={**self.component_results, **self.bus_results}[label].variables['charge_state'],
            mode='lines',
            name='Charge State',
        ))
        if show:
            plotly.offline.plot(fig)
        return fig

    def visualize_network(self,
                          path: Union[bool, str, pathlib.Path] = 'results/network.html',
                          controls: Union[bool, List[Literal[
                              'nodes', 'edges', 'layout', 'interaction', 'manipulation',
                              'physics', 'selection', 'renderer']]] = True,
                          show: bool = True
                          ) -> Optional['pyvis.network.Network']:
        """
        Visualizes the network structure of a FLowSystem using PyVis, saving it as an interactive HTML file.

        Parameters:
        - path (Union[bool, str, pathlib.Path], default='results/network.html'):
          Path to save the HTML visualization.
            - `False`: Visualization is created but not saved.
            - `str` or `Path`: Specifies file path (default: 'results/network.html').

        - controls (Union[bool, List[str]], default=True):
          UI controls to add to the visualization.
            - `True`: Enables all available controls.
            - `List`: Specify controls, e.g., ['nodes', 'layout'].
            - Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.

        - show (bool, default=True):
          Whether to open the visualization in the web browser.

        Returns:
        - Optional[pyvis.network.Network]: The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

        Usage:
        - Visualize and open the network with default options.
            self.visualize_network()

        - Save the visualization without opening:
            self.visualize_network(show=False)

        - Visualize with custom controls and path:
            self.visualize_network(path='output/custom_network.html', controls=['nodes', 'layout'])

        Notes:
        - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
        - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
        """
        from . import plotting
        return plotting.visualize_network(self.all_infos['Network']['Nodes'],
                                          self.all_infos['Network']['Edges'], path, controls, show)


class FlowResults(ElementResults):
    def __init__(self, infos: Dict, data: Dict, label_of_component: str) -> None:
        super().__init__(infos, data)
        self.is_input_in_component = self.all_infos['is_input_in_component']
        self.component_label = label_of_component
        self.bus_label = self.all_infos['bus']['label']
        self.label_full = f'{label_of_component}__{self.label}'
        self.variables = self.all_results

    def to_dataframe(self, variable_name: str = 'flow_rate') -> pd.DataFrame:
        return pd.DataFrame({variable_name: self.variables[variable_name]})


class ComponentResults(ElementResults):

    def __init__(self, infos: Dict, data: Dict):
        super().__init__(infos, data)
        inputs, outputs = self._create_flow_results()
        self.inputs: List[FlowResults] = inputs
        self.outputs: List[FlowResults] = outputs
        self.variables = {key: val for key, val in self.all_results.items() if key not in self.inputs + self.outputs}

    def _create_flow_results(self) -> Tuple[List[FlowResults], List[FlowResults]]:
        flow_infos = {flow['label']: flow for flow in self.all_infos['inputs'] + self.all_infos['outputs']}
        flow_results = {flow_info['label']: self.all_results[flow_info['label']] for flow_info in flow_infos.values()}
        flows = [FlowResults(flow_info, flow_result, self.label)
                 for flow_info, flow_result in zip(flow_infos.values(), flow_results.values())]
        inputs = [flow for flow in flows if flow.is_input_in_component]
        outputs = [flow for flow in flows if not flow.is_input_in_component]
        return inputs, outputs

    def to_dataframe(self,
                     variable_name: str = 'flow_rate',
                     input_factor: Optional[Literal[1, -1]] = -1,
                     output_factor: Optional[Literal[1, -1]] = 1) -> pd.DataFrame:
        inputs, outputs = {}, {}
        if input_factor is not None:
            inputs = {flow.label_full: (flow.variables[variable_name] * input_factor) for flow in self.inputs}
        if output_factor is not None:
            outputs = {flow.label_full: flow.variables[variable_name] * output_factor for flow in self.outputs}

        return pd.DataFrame(data={**inputs, **outputs})


class BusResults(ElementResults):
    def __init__(self, infos: Dict, data: Dict, inputs: List[FlowResults], outputs: List[FlowResults]):
        super().__init__(infos, data)
        self.inputs = inputs
        self.outputs = outputs
        self.variables = {key: val for key, val in self.all_results.items() if key not in self.inputs + self.outputs}

    def to_dataframe(self,
                     variable_name: str = 'flow_rate',
                     input_factor: Optional[Literal[1, -1]] = -1,
                     output_factor: Optional[Literal[1, -1]] = 1) -> pd.DataFrame:
        inputs, outputs = {}, {}
        if input_factor is not None:
            inputs = {flow.label_full: (flow.variables[variable_name] * input_factor) for flow in self.inputs}
            inputs['Excess Input'] = self.variables['excess_input'] * input_factor
        if output_factor is not None:
            outputs = {flow.label_full: flow.variables[variable_name] * output_factor for flow in self.outputs}
            outputs['Excess Output'] = self.variables['excess_output'] * output_factor

        return pd.DataFrame(data={**inputs, **outputs})


class EffectResults(ElementResults):
    pass


def extract_single_result(results_data: dict[str, Dict[str, Union[int, float, np.ndarray, dict]]],
                          keys: List[str]) -> Optional[Union[int, float, np.ndarray]]:
    """ Goes through a nested dictionary with the given keys. Returns the value if found. Else returns None"""
    for key in keys:
        if isinstance(results_data, dict):
            results_data = results_data.get(key, None)
        else:
            return None
    return results_data

def extract_results(results_data: dict[str, Dict[str, Union[int, float, np.ndarray, dict]]],
                    keys: List[str],
                    keep_none: bool = False) -> Dict[str, Union[int, float, np.ndarray]]:
    """ For each item in a dictionary, goes through its sub dictionaries.
    Returns the value if found. Else returns None. If specified, removes all None values
    """
    data = {kind: extract_single_result(results_data.get(kind, {}), keys) for kind in results_data.keys()}
    if keep_none:
        return data
    else:
        return {key:value for key, value in data.items() if value is not None}


if __name__ == '__main__':
    results = CalculationResults('Sim1', '/Users/felix/Documents/Dokumente - eigene/Neuer Ordner/flixOpt-Fork/examples/Ex02_complex/results')

    results.to_dataframe('Kessel')
    results.plot_flow_rate('Kessel__Q_fu', 'heatmap')
    plotting.heat_map_plotly(plotting.heat_map_data_from_df(
        pd.DataFrame(results.component_results['Speicher'].variables['charge_state'], index=results.time_with_end),
    periods='D', steps_per_period='15min'))

    results.plot_operation('Fernwärme', 'area', engine='plotly')
    fig = results.plot_operation('Fernwärme', 'area', engine='plotly')
    fig = plotting.with_plotly(results.to_dataframe('Wärmelast'), 'line', fig=fig)
    import plotly.offline
    plotly.offline.plot(fig)

    extract_results(results.all_results['Components'], ['Q_th', 'flow_rate'])
    extract_single_result(results.all_results['Components'], [ 'Kessel', 'Q_th', 'flow_rate'])

    fig = plotting.with_plotly(
        pd.DataFrame(extract_results(results.all_results['Components'], ['OnOff', 'on']), index=results.time),
                               mode='bar')
    fig.update_layout(barmode='group', bargap=0.2, bargroupgap = 0.1)
    plotly.offline.plot(fig)

    print()
