import unittest

import numpy as np
import datetime

from flixOpt.flixComps import cKessel
from flixOpt.flixStructure import cFlow, cBus, cEnergySystem


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
        #self.assertEqual(exists, kessel.exists.d, msg=f"Kessel exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, kessel.inputs[0].exists.d),
                        msg=f"Kessel input exists mismatch: Expected {exists}, got {kessel.exists}")
        self.assertTrue(np.array_equal(exists, kessel.outputs[0].exists.d_i),
                         msg=f"Kessel output exists mismatch: Expected {exists}, got {kessel.exists}")



if __name__ == '__main__':
    unittest.main()
