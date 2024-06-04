import unittest

import numpy as np
import datetime

from flixOpt.flixComps import cKessel, cStorage
from flixOpt.flixStructure import cFlow, cBus, cEnergySystem, cCalculation, cEffectType
from flixOpt.flixPostprocessing import flix_results


class TestExistance(unittest.TestCase):
    def test_flow(self):
        exists = np.array([1,1,1,1,0,0,0,0])
        kessel = cKessel(label="Kessel",
                         eta=0.5,
                         exists=exists,
                         Q_th=cFlow(label="Q_th",
                                    medium="media",
                                    bus=cBus(label="bus1", media="media")),
                         Q_fu=cFlow(label="Q_fu",
                                    medium="media",
                                     bus=cBus(label="bus2", media="media"))
                         )
        aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(exists)) * datetime.timedelta(
            hours=1)  # creating timeseries
        aTimeSeries = aTimeSeries.astype('datetime64')  # needed format for timeseries in flixOpt
        es = cEnergySystem(aTimeSeries)
        es.addComponents(kessel)
        es.addEffects(cEffectType(label="costs", unit="€", isStandard=True, isObjective=True, description=""))
        calc = cCalculation('Sim1', es, 'pyomo')
        calc.doModelingAsOneSegment()
        calc.solve(solverProps={'gapFrac': 0.05,
                       'timelimit': 60,
                       'solver': 'cbc',
                       'displaySolverOutput': True,
                       })
        # self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, kessel.inputs[0].exists_with_comp.d_i))
        self.assertTrue(np.array_equal(exists, kessel.outputs[0].exists_with_comp.d_i))
        self.assertTrue(np.array_equal(exists, kessel.inputs[0].max_rel_with_exists.d_i))
        self.assertTrue(np.array_equal(exists, kessel.outputs[0].max_rel_with_exists.d_i))

    def test_storage(self):
        exists = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        comp = cStorage(label="Speicher",
                        exists=exists,
                        capacity_inFlowHours=100,
                        inFlow=cFlow(label="in",
                                     nominal_val=5,
                                     bus=cBus(label="bus1", media="media")),
                        outFlow=cFlow(label="out",
                                      nominal_val=5,
                                      bus=cBus(label="bus2", media="media"))
                        )
        aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(exists)) * datetime.timedelta(
            hours=1)  # creating timeseries
        aTimeSeries = aTimeSeries.astype('datetime64')  # needed format for timeseries in flixOpt
        es = cEnergySystem(aTimeSeries)
        es.addComponents(comp)
        es.addEffects(cEffectType(label="costs", unit="€", isStandard=True, isObjective=True, description=""))
        calc = cCalculation('Sim1', es, 'pyomo')
        calc.doModelingAsOneSegment()
        calc.solve(solverProps={'gapFrac': 0.05,
                                'timelimit': 60,
                                'solver': 'cbc',
                                'displaySolverOutput': True,
                                })
        # self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, comp.inputs[0].exists_with_comp.d_i))
        self.assertTrue(np.array_equal(exists, comp.outputs[0].exists_with_comp.d_i))
        self.assertTrue(np.array_equal(exists, comp.inputs[0].max_rel_with_exists.d_i))
        self.assertTrue(np.array_equal(exists, comp.outputs[0].max_rel_with_exists.d_i))


if __name__ == '__main__':
    unittest.main()
