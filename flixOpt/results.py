# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:05:50 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
import json
import pathlib
from typing import Dict, List, Tuple

import yaml

logger = logging.getLogger('flixOpt')


class ElementResults:
    def __init__(self, infos: Dict, data: Dict):
        self._infos = infos
        self._data = data
        self.label = self._infos['label']

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label})'


class CalculationResults:
    def __init__(self, calculation_name: str, folder: str) -> None:
        self._path_infos = (pathlib.Path(folder) / f'{calculation_name}_info.yaml').resolve().as_posix()
        self._path_results = (pathlib.Path(folder) / f'{calculation_name}_data.json').resolve().as_posix()

        with open(self._path_infos, 'rb') as f:
            self.infos: Dict = yaml.safe_load(f)

        with open(self._path_results, 'rb') as f:
            self.results: Dict = json.load(f)

        self.component_results: Dict[str, ComponentResults] = {}
        self.effect_results: Dict[str, EffectResults] = {}
        self.bus_results: Dict[str, BusResults] = {}

        self._construct_component_results()
        self._construct_bus_results()
        self._construct_effect_results()

    def _construct_component_results(self):
        comp_results = self.results['Components']
        comp_infos = self.infos['FlowSystem']['Components']
        assert comp_results.keys() == comp_infos.keys(), \
            f'Missing Component or mismatched keys: {comp_results.keys() ^ comp_infos.keys()}'

        for key in comp_results.keys():
            infos, results = comp_infos[key], comp_results[key]
            res = ComponentResults(infos, results)
            self.component_results[res.label] = res

    def _construct_effect_results(self):
        effect_results = self.results['Effects']
        effect_infos = self.infos['FlowSystem']['Effects']
        effect_infos['penalty'] = {'label': 'Penalty'}
        assert effect_results.keys() == effect_infos.keys(), \
            f'Missing Effect or mismatched keys: {effect_results.keys() ^ effect_infos.keys()}'

        for key in effect_results.keys():
            infos, results = effect_infos[key], effect_results[key]
            res = EffectResults(infos, results)
            self.effect_results[res.label] = res

    def _construct_bus_results(self):
        """ This has to be called afte _construct_component_results(), as its"""
        bus_results = self.results['Buses']
        bus_infos = self.infos['FlowSystem']['Buses']
        assert bus_results.keys() == bus_infos.keys(), \
            f'Missing Bus or mismatched keys: {bus_results.keys() ^ bus_infos.keys()}'

        for key in bus_results.keys():
            infos, results = bus_infos[key], bus_results[key]
            inputs = [flow for flow in self.flow_results().values() if not flow.is_input_in_component]
            outputs = [flow for flow in self.flow_results().values() if flow.is_input_in_component]
            res = BusResults(infos, results, inputs, outputs)
            self.bus_results[res.label] = res

    def flow_results(self) -> Dict[str, 'FlowResults']:
        return {flow.label_full: flow
                for comp in self.component_results.values()
                for flow in comp.inputs + comp.outputs}


class FlowResults(ElementResults):
    def __init__(self, infos: Dict, data: Dict, label_of_component: str) -> None:
        super().__init__(infos, data)
        self.is_input_in_component = self._infos['is_input_in_component']
        self.label_of_component = label_of_component
        self.label_full = f'{label_of_component}__{self.label}'
        self.variables = self._data


class ComponentResults(ElementResults):

    def __init__(self, infos: Dict, data: Dict):
        super().__init__(infos, data)
        inputs, outputs = self._create_flow_results()
        self.inputs: List[FlowResults] = inputs
        self.outputs: List[FlowResults] = outputs
        self.variables = {key: val for key, val in self._data.items() if key not in self.inputs + self.outputs}

    def _create_flow_results(self) -> Tuple[List[FlowResults], List[FlowResults]]:
        flow_infos = {key: value for key, value in self._infos.items() if
                      isinstance(value, dict) and 'Flow' in value.get('class', '')}
        flow_results = {flow_info['label']: self._data[flow_info['label']] for flow_info in flow_infos.values()}
        flows = [FlowResults(flow_info, flow_result, self.label)
                 for flow_info, flow_result in zip(flow_infos.values(), flow_results.values())]
        inputs = [flow for flow in flows if flow.is_input_in_component]
        outputs = [flow for flow in flows if not flow.is_input_in_component]
        return inputs, outputs


class BusResults(ElementResults):
    def __init__(self, infos: Dict, data: Dict, inputs, outputs):
        super().__init__(infos, data)
        self.inputs = inputs
        self.outputs = outputs
        self.variables = {key: val for key, val in self._data.items() if key not in self.inputs + self.outputs}


class EffectResults(ElementResults):
    pass


if __name__ == '__main__':
    results = CalculationResults('Sim1', '/Users/felix/Documents/Dokumente - eigene/Neuer Ordner/flixOpt-Fork/examples/Ex01_simple/results')

    print()
