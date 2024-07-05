import unittest

import numpy as np
import datetime

from flixOpt.components import Boiler, Storage, Source, Sink, CHP
from flixOpt.structure import Flow, Bus, System, Calculation, Effect
from flixOpt.flixPostprocessing import flix_results


class TestExistance(unittest.TestCase):
    def setUp(self):
        aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(5) * datetime.timedelta(hours=1)
        self.es = System(aTimeSeries.astype('datetime64'))
        self.effects = {
            "costs": Effect(label="costs", unit="€", is_standard=True, is_objective=True, description="")
        }
        self.busses = {
            "Gas": Bus(label="Gas", media="fuel"),
            "Heat": Bus(label="Heat", media="heat"),
            "Power": Bus(label="Power", media="el")
        }
        self.sinks_n_sources = {
            "GasSource": Source(label="GasSource", source=Flow(label="Gasmarkt", bus=self.busses["Gas"],
                                                               effects_per_flow_hour=np.array([20, 15, 13, 25, 26]))),
            "HeatSink": Sink(label="HeatSink", sink=Flow(label="Heating_Network",
                                                         size=1,
                                                         val_rel= np.linspace(0, 100, len(self.es.time_series)),
                                                         bus=self.busses["Heat"])),
            "PowerSource": Source(label="PowerSource", source=Flow(label="Power_Grid", bus=self.busses["Power"],
                                                                   effects_per_flow_hour=np.array([100, 20, 60, 40, 5])))
        }
        self.comps = {
            "Boiler": Boiler(label="Boiler", eta=0.5,
                             Q_th=Flow(label="Q_th", size=112, bus=self.busses["Heat"]),
                             Q_fu=Flow(label="Q_fu", bus=self.busses["Gas"])),
            "CHP": CHP(label="CHP", eta_th=0.45, eta_el=0.4,
                       Q_th=Flow(label="Q_th", size=112, bus=self.busses["Heat"]),
                       P_el=Flow(label="P_el", bus=self.busses["Power"]),
                       Q_fu=Flow(label="Q_fu", bus=self.busses["Gas"])
                       )
        }

    def test_boiler(self):
        exists = np.array([1, 1, 1, 0, 0])
        size=120
        boiler_exists = Boiler(label="Boiler_ex", eta=0.8, exists= exists,
                               Q_th=Flow(label="Q_th", size=size, bus=self.busses["Heat"]),
                               Q_fu=Flow(label="Q_fu", bus=self.busses["Gas"])
                               )
        self.es.add_elements(*self.effects.values(), *self.sinks_n_sources.values())
        self.es.add_elements(boiler_exists, self.comps["Boiler"])
        calc = Calculation('Sim1', self.es, 'pyomo')
        calc.do_modeling_as_one_segment()
        calc.solve(solverProps={'mip_gap': 0.05,
                                'time_limit_seconds': 60,
                                'solver_name': 'cbc',
                                'solver_output_to_console': True,
                                })
        # self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, boiler_exists.inputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, boiler_exists.outputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, boiler_exists.inputs[0].max_rel_with_exists.active_data))
        self.assertTrue(np.array_equal(exists, boiler_exists.outputs[0].max_rel_with_exists.active_data))

        results = flix_results(calc.nameOfCalc).results
        self.assertTrue(np.all(results["Boiler_ex"]["Q_th"]["val"] <= exists * size))

    def test_storage(self):
        exists = np.array([0, 0, 1, 1, 1])
        size=5
        capacity = 10
        storage_exists = Storage(label="Storage_ex", exists=exists, capacity_inFlowHours=capacity,
                                 inFlow=Flow(label="in",
                                             size=size,
                                             bus=self.busses["Gas"]),
                                 outFlow=Flow(label="out",
                                              size=size,
                                              bus=self.busses["Gas"])
                                 )

        self.es.add_elements(*self.effects.values(), *self.sinks_n_sources.values())
        self.es.add_elements(storage_exists, self.comps["CHP"])
        calc = Calculation('Sim1', self.es, 'pyomo')
        calc.do_modeling_as_one_segment()
        calc.solve(solverProps={'mip_gap': 0.05,
                                'time_limit_seconds': 60,
                                'solver_name': 'cbc',
                                'solver_output_to_console': True,
                                })
        # self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, storage_exists.inputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.outputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.inputs[0].max_rel_with_exists.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.outputs[0].max_rel_with_exists.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.max_rel_chargeState.active_data))

        results = flix_results(calc.nameOfCalc).results
        self.assertTrue(np.all(results["Storage_ex"]["in"]["val"] <= exists * size))
        self.assertTrue(np.all(results["Storage_ex"]["out"]["val"] <= exists * size))
        self.assertTrue(np.all(results["Storage_ex"]["charge_state"] <= np.append(exists * capacity, exists[-1] * capacity)))

    def tearDown(self):
        self.es = None
        self.effects = None
        self.busses = None
        self.comps = None
        self.sinks_n_sources = None


if __name__ == '__main__':
    unittest.main()
