import unittest

import numpy as np
import datetime

from flixOpt.flixComps import Boiler, Storage, Source, Sink, CHP
from flixOpt.flixStructure import Flow, Bus, System, Calculation, Effect
from flixOpt.flixPostprocessing import flix_results


class TestExistance(unittest.TestCase):
    def setUp(self):
        aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(5) * datetime.timedelta(hours=1)
        self.es = System(aTimeSeries.astype('datetime64'))
        self.effects = {
            "costs": Effect(label="costs", unit="€", isStandard=True, isObjective=True, description="")
        }
        self.busses = {
            "Gas": Bus(label="Gas", media="fuel"),
            "Heat": Bus(label="Heat", media="heat"),
            "Power": Bus(label="Power", media="el")
        }
        self.sinks_n_sources = {
            "GasSource": Source(label="GasSource", source=Flow(label="Gasmarkt", bus=self.busses["Gas"],
                                                               costsPerFlowHour=np.array([20, 15, 13, 25, 26]))),
            "HeatSink": Sink(label="HeatSink", sink=Flow(label="Heating_Network",
                                                         nominal_val=1,
                                                         val_rel= np.linspace(0, 100, len(self.es.timeSeries)),
                                                         bus=self.busses["Heat"])),
            "PowerSource": Source(label="PowerSource", source=Flow(label="Power_Grid", bus=self.busses["Power"],
                                                                   costsPerFlowHour=np.array([100, 20, 60, 40, 5])))
        }
        self.comps = {
            "Boiler": Boiler(label="Boiler", eta=0.5,
                             Q_th=Flow(label="Q_th", nominal_val=112, bus=self.busses["Heat"]),
                             Q_fu=Flow(label="Q_fu", bus=self.busses["Gas"])),
            "CHP": CHP(label="CHP", eta_th=0.45, eta_el=0.4,
                       Q_th=Flow(label="Q_th", nominal_val=112, bus=self.busses["Heat"]),
                       P_el=Flow(label="P_el", bus=self.busses["Power"]),
                       Q_fu=Flow(label="Q_fu", bus=self.busses["Gas"])
                       )
        }

    def test_boiler(self):
        exists = np.array([1, 1, 1, 0, 0])
        nominal_val = 120
        boiler_exists = Boiler(label="Boiler_ex", eta=0.8, exists= exists,
                               Q_th=Flow(label="Q_th", nominal_val=nominal_val, bus=self.busses["Heat"]),
                               Q_fu=Flow(label="Q_fu", bus=self.busses["Gas"])
                               )
        self.es.addElements(*self.effects.values(), *self.sinks_n_sources.values())
        self.es.addElements(boiler_exists, self.comps["Boiler"])
        calc = Calculation('Sim1', self.es, 'pyomo')
        calc.doModelingAsOneSegment()
        calc.solve(solverProps={'gapFrac': 0.05,
                                'time_limit_seconds': 60,
                                'solver': 'cbc',
                                'solver_output_to_console': True,
                                })
        # self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, boiler_exists.inputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, boiler_exists.outputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, boiler_exists.inputs[0].max_rel_with_exists.active_data))
        self.assertTrue(np.array_equal(exists, boiler_exists.outputs[0].max_rel_with_exists.active_data))

        results = flix_results(calc.nameOfCalc).results
        self.assertTrue(np.all(results["Boiler_ex"]["Q_th"]["val"] <= exists * nominal_val))

    def test_storage(self):
        exists = np.array([0, 0, 1, 1, 1])
        nominal_val = 5
        capacity = 10
        storage_exists = Storage(label="Storage_ex", exists=exists, capacity_inFlowHours=capacity,
                                 inFlow=Flow(label="in",
                                             nominal_val=nominal_val,
                                             bus=self.busses["Gas"]),
                                 outFlow=Flow(label="out",
                                              nominal_val=nominal_val,
                                              bus=self.busses["Gas"])
                                 )

        self.es.addElements(*self.effects.values(), *self.sinks_n_sources.values())
        self.es.addElements(storage_exists, self.comps["CHP"])
        calc = Calculation('Sim1', self.es, 'pyomo')
        calc.doModelingAsOneSegment()
        calc.solve(solverProps={'gapFrac': 0.05,
                                'time_limit_seconds': 60,
                                'solver': 'cbc',
                                'solver_output_to_console': True,
                                })
        # self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, storage_exists.inputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.outputs[0].exists_with_comp.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.inputs[0].max_rel_with_exists.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.outputs[0].max_rel_with_exists.active_data))
        self.assertTrue(np.array_equal(exists, storage_exists.max_rel_chargeState.active_data))

        results = flix_results(calc.nameOfCalc).results
        self.assertTrue(np.all(results["Storage_ex"]["in"]["val"] <= exists * nominal_val))
        self.assertTrue(np.all(results["Storage_ex"]["out"]["val"] <= exists * nominal_val))
        self.assertTrue(np.all(results["Storage_ex"]["charge_state"] <= np.append(exists * capacity, exists[-1] * capacity)))

    def tearDown(self):
        self.es = None
        self.effects = None
        self.busses = None
        self.comps = None
        self.sinks_n_sources = None


if __name__ == '__main__':
    unittest.main()
