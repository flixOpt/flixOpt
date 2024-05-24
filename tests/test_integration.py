import unittest
import numpy as np
import pandas as pd
import os
import datetime
from flixOpt.flixStructure import *
from flixOpt.flixComps import *
import flixOpt.flixPostprocessing as flixPost

class BaseTest(unittest.TestCase):
    def setUp(self):
        self.solverProps = {
            'gapFrac': 0.0001,
            'timelimit': 3600,
            'solver': 'cbc',
            'displaySolverOutput': True,
        }

    def assertAlmostEqualNumeric(self, actual, desired, err_msg):
        '''
        Asserts that actual is almost equal to desired.
        Designed for comparing float and ndarrays. Whith respect to tolerances
        '''
        relative_error_range_in_percent = 0.01
        relative_tol = relative_error_range_in_percent/100
        if isinstance(desired, (int, float)):
            delta = abs(relative_tol * desired)
            self.assertAlmostEqual(actual, desired, msg=err_msg, delta=delta)
        else:
            np.testing.assert_allclose(actual, desired, rtol=relative_tol, atol=1e-9)


class TestSimple(BaseTest):

    def setUp(self):
        super().setUp()

        self.Q_th_Last = np.array([30., 0., 90., 110, 110, 20, 20, 20, 20])
        self.p_el = 1 / 1000 * np.array([80., 80., 80., 80, 80, 80, 80, 80, 80])
        self.aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(hours=1)
        self.aTimeSeries = self.aTimeSeries.astype('datetime64')
        self.max_emissions_per_hour = 1000

    def test_model(self):
        results = self.model()

        # Compare expected values with actual values
        self.assertAlmostEqualNumeric(results['costs']['all']['sum'], 81.88394666666667,
                               "costs doesnt match expected value")
        self.assertAlmostEqualNumeric(results['CO2']['all']['sum'], 255.09184,
                               "CO2 doesnt match expected value")
        self.assertAlmostEqualNumeric(results['Boiler']['Q_th']['val'],
                                      [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
                                  "Q_th doesnt match expected value")
        self.assertAlmostEqualNumeric(results['CHP_unit']['Q_th']['val'],
                                      [30., 26.66666667, 75., 75., 75., 20., 20., 20., 20.],
                                  "Q_th doesnt match expected value")

    def model(self):
        # Define the components and energy system
        Strom = cBus('el', 'Strom')
        Fernwaerme = cBus('heat', 'Fernwärme')
        Gas = cBus('fuel', 'Gas')

        costs = cEffectType('costs', '€', 'Kosten', isStandard=True, isObjective=True)
        CO2 = cEffectType('CO2', 'kg', 'CO2_e-Emissionen', specificShareToOtherEffects_operation={costs: 0.2},
                          max_per_hour_operation=self.max_emissions_per_hour)

        aBoiler = cKessel('Boiler', eta=0.5,
                          Q_th=cFlow('Q_th', bus=Fernwaerme, nominal_val=50, min_rel=5 / 50, max_rel=1),
                          Q_fu=cFlow('Q_fu', bus=Gas))
        aKWK = cKWK('CHP_unit', eta_th=0.5, eta_el=0.4, P_el=cFlow('P_el', bus=Strom, nominal_val=60, min_rel=5 / 60),
                    Q_th=cFlow('Q_th', bus=Fernwaerme), Q_fu=cFlow('Q_fu', bus=Gas))
        aSpeicher = cStorage('Speicher', inFlow=cFlow('Q_th_load', bus=Fernwaerme, nominal_val=1e4),
                             outFlow=cFlow('Q_th_unload', bus=Fernwaerme, nominal_val=1e4), capacity_inFlowHours=30,
                             chargeState0_inFlowHours=0,
                             max_rel_chargeState=1 / 100 * np.array([80., 70., 80., 80, 80, 80, 80, 80, 80, 80]),
                             eta_load=0.9, eta_unload=1, fracLossPerHour=0.08, avoidInAndOutAtOnce=True,
                             investArgs=cInvestArgs(fixCosts=20, investmentSize_is_fixed=True,
                                                    investment_is_optional=False))
        aWaermeLast = cSink('Wärmelast', sink=cFlow('Q_th_Last', bus=Fernwaerme, nominal_val=1, val_rel=self.Q_th_Last))
        aGasTarif = cSource('Gastarif',
                            source=cFlow('Q_Gas', bus=Gas, nominal_val=1000, costsPerFlowHour={costs: 0.04, CO2: 0.3}))
        aStromEinspeisung = cSink('Einspeisung', sink=cFlow('P_el', bus=Strom, costsPerFlowHour=-1 * self.p_el))

        es = cEnergySystem(self.aTimeSeries, dt_last=None)
        es.addComponents(aSpeicher)
        es.addEffects(costs, CO2)
        es.addComponents(aBoiler, aWaermeLast, aGasTarif)
        es.addComponents(aStromEinspeisung)
        es.addComponents(aKWK)

        chosenEsTimeIndexe = None

        aCalc = cCalculation('Test_Sim', es, 'pyomo', chosenEsTimeIndexe)
        aCalc.doModelingAsOneSegment()

        es.printModel()
        es.printVariables()
        es.printEquations()

        aCalc.solve(self.solverProps, nameSuffix='_highs')

        nameOfCalc = aCalc.nameOfCalc
        aCalc_post = flixPost.flix_results(nameOfCalc)
        return aCalc_post.results


class TestComplex(BaseTest):

    def setUp(self):
        super().setUp()
        self.Q_th_Last = np.array([30., 0., 90., 110, 110, 20, 20, 20, 20])
        self.P_el_Last = np.array([40., 40., 40., 40 , 40, 40, 40, 40, 40])
        self.aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(hours=1)
        self.aTimeSeries = self.aTimeSeries.astype('datetime64')
        self.excessCosts = None
        self.useCHPwithLinearSegments = False

    def test_basic(self):
        results = self.basic_model()

        # Compare expected values with actual values
        self.assertAlmostEqualNumeric(results['costs']['all']['sum'], -11597.873624489237,
                               "costs doesnt match expected value")
        self.assertAlmostEqualNumeric(results['CO2']['all']['sum'], 1294.186483480967,
                               "CO2 doesnt match expected value")
        self.assertAlmostEqualNumeric(results['Kessel']['Q_th']['val'],
                                      [0, 0, 0, 45, 0, 0, 0, 0, 0],
                                  "Kessel doesnt match expected value")
        self.assertAlmostEqualNumeric(results['KWK']['Q_th']['val'],
                                      [7.50000000e+01, 6.97111111e+01, 7.50000000e+01, 7.50000000e+01,
                                   7.39330280e+01, 7.50000000e+01, 0.00000000e+00, 3.12638804e-14,
                                   3.83693077e-14],
                                  "KWK Q_th doesnt match expected value")
        self.assertAlmostEqualNumeric(results['KWK']['P_el']['val'],
                                      [6.00000000e+01, 5.57688889e+01, 6.00000000e+01, 6.00000000e+01,
                                   5.91464224e+01, 6.00000000e+01, 0.00000000e+00, 2.50111043e-14, 3.06954462e-14],
                                  "KWK P_el doesnt match expected value")

        self.assertAlmostEqualNumeric(results['KWK']['P_el']['val'],
                                      [6.00000000e+01, 5.57688889e+01, 6.00000000e+01, 6.00000000e+01,
                                   5.91464224e+01, 6.00000000e+01, 0.00000000e+00, 2.50111043e-14, 3.06954462e-14],
                                  "KWK P_el doesnt match expected value")

        self.assertAlmostEqualNumeric(results['Speicher']['nettoFlow'],
                                      [-45., -69.71111111, 15., -10., 36.06697198, -55., 20., 20., 20.],
                                  "Speicher nettoFlow doesnt match expected value")

        self.assertAlmostEqualNumeric(results['Speicher']['invest']['investCosts_segmented_costs'], 800,
                                  "Speicher investCosts_segmented_costs doesnt match expected value")

    def test_segments_of_flows(self):
        results = self.segments_of_flows_model()

        # Compare expected values with actual values
        self.assertAlmostEqualNumeric(results['costs']['all']['sum'], -10710.997365760755,
                               "costs doesnt match expected value")
        self.assertAlmostEqualNumeric(results['CO2']['all']['sum'], 1278.7939026086956,
                               "CO2 doesnt match expected value")
        self.assertAlmostEqualNumeric(results['Kessel']['Q_th']['val'],
                                      [0, 0, 0, 45, 0, 0, 0, 0, 0],
                                  "Kessel doesnt match expected value")
        self.assertAlmostEqualNumeric(results['KWK']['Q_th']['val'],
                                      [45., 45., 64.5962087, 100.,
                                       61.3136, 45., 45., 12.86469565,
                                       0.],
                                  "KWK Q_th doesnt match expected value")
        self.assertAlmostEqualNumeric(results['KWK']['P_el']['val'],
                                      [40., 40., 47.12589407, 60., 45.93221818,
                                       40., 40., 10.91784108, -0.],
                                  "KWK P_el doesnt match expected value")

        self.assertAlmostEqualNumeric(results['Speicher']['nettoFlow'],
                                      [-15., -45., 25.4037913, -35.,
                                       48.6864, -25., -25., 7.13530435,
                                       20.],
                                  "Speicher nettoFlow doesnt match expected value")

        self.assertAlmostEqualNumeric(results['Speicher']['invest']['investCosts_segmented_costs'], 454.74666666666667,
                                  "Speicher investCosts_segmented_costs doesnt match expected value")

    def basic_model(self):
        # Define the components and energy system
        Strom = cBus('el', 'Strom', excessCostsPerFlowHour=self.excessCosts)
        Fernwaerme = cBus('heat', 'Fernwärme', excessCostsPerFlowHour=self.excessCosts)
        Gas = cBus('fuel', 'Gas', excessCostsPerFlowHour=self.excessCosts)

        costs = cEffectType('costs', '€', 'Kosten', isStandard=True, isObjective=True)
        CO2 = cEffectType('CO2', 'kg', 'CO2_e-Emissionen', specificShareToOtherEffects_operation={costs: 0.2})
        PE = cEffectType('PE', 'kWh_PE', 'Primärenergie', max_Sum=3.5e3)

        invest_Gaskessel = cInvestArgs(fixCosts=1000, investmentSize_is_fixed=True, investment_is_optional=False, specificCosts={costs: 10, PE: 2})
        aGaskessel = cKessel('Kessel', eta=0.5, costsPerRunningHour={costs: 0, CO2: 1000},
                             Q_th=cFlow('Q_th', bus=Fernwaerme, nominal_val=50, loadFactor_max=1.0, loadFactor_min=0.1, min_rel=5 / 50, max_rel=1, onHoursSum_min=0, onHoursSum_max=1000, onHours_max=10, offHours_max=10, switchOnCosts=0.01, switchOn_maxNr=1000, valuesBeforeBegin=[50], investArgs=invest_Gaskessel, sumFlowHours_max=1e6),
                             Q_fu=cFlow('Q_fu', bus=Gas, nominal_val=200, min_rel=0, max_rel=1))

        aKWK = cKWK('KWK', eta_th=0.5, eta_el=0.4, switchOnCosts=0.01,
                    P_el=cFlow('P_el', bus=Strom, nominal_val=60, min_rel=5 / 60),
                    Q_th=cFlow('Q_th', bus=Fernwaerme, nominal_val=1e3),
                    Q_fu=cFlow('Q_fu', bus=Gas, nominal_val=1e3), on_valuesBeforeBegin=[1])

        costsInvestsizeSegments = [[5, 25, 25, 100], {costs: [50, 250, 250, 800], PE: [5, 25, 25, 100]}]
        invest_Speicher = cInvestArgs(fixCosts=0, investmentSize_is_fixed=False, costsInInvestsizeSegments=costsInvestsizeSegments, investment_is_optional=False, specificCosts={costs: 0.01, CO2: 0.01}, min_investmentSize=0, max_investmentSize=1000)
        aSpeicher = cStorage('Speicher', inFlow=cFlow('Q_th_load', bus=Fernwaerme, nominal_val=1e4), outFlow=cFlow('Q_th_unload', bus=Fernwaerme, nominal_val=1e4), capacity_inFlowHours=None, chargeState0_inFlowHours=0, charge_state_end_max=10, eta_load=0.9, eta_unload=1, fracLossPerHour=0.08, avoidInAndOutAtOnce=True, investArgs=invest_Speicher)

        aWaermeLast = cSink('Wärmelast', sink=cFlow('Q_th_Last', bus=Fernwaerme, nominal_val=1, min_rel=0, val_rel=self.Q_th_Last))
        aGasTarif = cSource('Gastarif', source=cFlow('Q_Gas', bus=Gas, nominal_val=1000, costsPerFlowHour={costs: 0.04, CO2: 0.3}))
        aStromEinspeisung = cSink('Einspeisung', sink=cFlow('P_el', bus=Strom, costsPerFlowHour=-1 * np.array(self.P_el_Last)))

        es = cEnergySystem(self.aTimeSeries, dt_last=None)
        es.addEffects(costs, CO2, PE)
        es.addComponents(aGaskessel, aWaermeLast, aGasTarif, aStromEinspeisung, aKWK, aSpeicher)

        aCalc = cCalculation('Sim1', es, 'pyomo', None)
        aCalc.doModelingAsOneSegment()

        es.printModel()
        es.printVariables()
        es.printEquations()

        aCalc.solve(self.solverProps, nameSuffix=f"_{self.solverProps['solver']}")

        return flixPost.flix_results(aCalc.nameOfCalc).results

    def segments_of_flows_model(self):
        # Define the components and energy system
        Strom = cBus('el', 'Strom', excessCostsPerFlowHour=self.excessCosts)
        Fernwaerme = cBus('heat', 'Fernwärme', excessCostsPerFlowHour=self.excessCosts)
        Gas = cBus('fuel', 'Gas', excessCostsPerFlowHour=self.excessCosts)

        costs = cEffectType('costs', '€', 'Kosten', isStandard=True, isObjective=True)
        CO2 = cEffectType('CO2', 'kg', 'CO2_e-Emissionen', specificShareToOtherEffects_operation={costs: 0.2})
        PE = cEffectType('PE', 'kWh_PE', 'Primärenergie', max_Sum=3.5e3)

        invest_Gaskessel = cInvestArgs(fixCosts=1000, investmentSize_is_fixed=True, investment_is_optional=False, specificCosts={costs: 10, PE: 2})
        aGaskessel = cKessel('Kessel', eta=0.5, costsPerRunningHour={costs: 0, CO2: 1000},
                             Q_th=cFlow('Q_th', bus=Fernwaerme, nominal_val=50, loadFactor_max=1.0, loadFactor_min=0.1, min_rel=5 / 50, max_rel=1, onHoursSum_min=0, onHoursSum_max=1000, onHours_max=10, offHours_max=10, switchOnCosts=0.01, switchOn_maxNr=1000, valuesBeforeBegin=[50], investArgs=invest_Gaskessel, sumFlowHours_max=1e6),
                             Q_fu=cFlow('Q_fu', bus=Gas, nominal_val=200, min_rel=0, max_rel=1))

        P_el = cFlow('P_el', bus=Strom, nominal_val=60, max_rel=55)
        Q_th = cFlow('Q_th', bus=Fernwaerme)
        Q_fu = cFlow('Q_fu', bus=Gas)
        segmentsOfFlows = {P_el: [5, 30, 40, 60], Q_th: [6, 35, 45, 100], Q_fu: [12, 70, 90, 200]}
        aKWK = cBaseLinearTransformer('KWK', inputs=[Q_fu], outputs=[P_el, Q_th], segmentsOfFlows=segmentsOfFlows, switchOnCosts=0.01, on_valuesBeforeBegin=[1])

        costsInvestsizeSegments = [[5, 25, 25, 100], {costs: [50, 250, 250, 800], PE: [5, 25, 25, 100]}]
        invest_Speicher = cInvestArgs(fixCosts=0, investmentSize_is_fixed=False, costsInInvestsizeSegments=costsInvestsizeSegments, investment_is_optional=False, specificCosts={costs: 0.01, CO2: 0.01}, min_investmentSize=0, max_investmentSize=1000)
        aSpeicher = cStorage('Speicher', inFlow=cFlow('Q_th_load', bus=Fernwaerme, nominal_val=1e4), outFlow=cFlow('Q_th_unload', bus=Fernwaerme, nominal_val=1e4), capacity_inFlowHours=None, chargeState0_inFlowHours=0, charge_state_end_max=10, eta_load=0.9, eta_unload=1, fracLossPerHour=0.08, avoidInAndOutAtOnce=True, investArgs=invest_Speicher)

        aWaermeLast = cSink('Wärmelast', sink=cFlow('Q_th_Last', bus=Fernwaerme, nominal_val=1, min_rel=0, val_rel=self.Q_th_Last))
        aGasTarif = cSource('Gastarif', source=cFlow('Q_Gas', bus=Gas, nominal_val=1000, costsPerFlowHour={costs: 0.04, CO2: 0.3}))
        aStromEinspeisung = cSink('Einspeisung', sink=cFlow('P_el', bus=Strom, costsPerFlowHour=-1 * np.array(self.P_el_Last)))

        es = cEnergySystem(self.aTimeSeries, dt_last=None)
        es.addEffects(costs, CO2, PE)
        es.addComponents(aGaskessel, aWaermeLast, aGasTarif, aStromEinspeisung, aKWK)
        es.addComponents(aSpeicher)

        aCalc = cCalculation('Sim1', es, 'pyomo', None)
        aCalc.doModelingAsOneSegment()

        es.printModel()
        es.printVariables()
        es.printEquations()

        aCalc.solve(self.solverProps, nameSuffix=f"_{self.solverProps['solver']}")

        return flixPost.flix_results(aCalc.nameOfCalc).results


class TestModelingTypes(BaseTest):

    def setUp(self):
        super().setUp()
        self.Q_th_Last = np.array([30., 0., 90., 110, 110, 20, 20, 20, 20])
        self.p_el = 1 / 1000 * np.array([80., 80., 80., 80, 80, 80, 80, 80, 80])
        self.aTimeSeries = (datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(hours=1)).astype('datetime64')
        self.max_emissions_per_hour = 1000

    def test_full(self):
        results = self.modeling_types("full")
        self.assertAlmostEqualNumeric(results['costs']['all']['sum'], 343613, "costs doesnt match expected value")

    def test_aggregated(self):
        results = self.modeling_types("aggregated")
        self.assertAlmostEqualNumeric(results['costs']['all']['sum'], 342967.0, "costs doesnt match expected value")

    def test_segmented(self):
        results = self.modeling_types("segmented")
        self.assertAlmostEqualNumeric(sum(results['costs']['operation']['sum_TS']), 343613, "costs doesnt match expected value")

    def modeling_types(self, modeling_type: Literal["full", "segmented", "aggregated"]):
        doFullCalc, doSegmentedCalc, doAggregatedCalc = modeling_type == "full", modeling_type == "segmented", modeling_type == "aggregated"
        if not any([doFullCalc, doSegmentedCalc, doAggregatedCalc]): raise Exception("Unknown modeling type")

        filename = os.path.join(os.path.dirname(__file__), "ressources", "Zeitreihen2020.csv")
        ts_raw = pd.read_csv(filename, index_col=0).sort_index()
        data = ts_raw['2020-01-01 00:00:00':'2020-12-31 23:45:00']['2020-01-01':'2020-01-03 23:45:00']
        P_el_Last, Q_th_Last, p_el, gP = data['P_Netz/MW'], data['Q_Netz/MW'], data['Strompr.€/MWh'], data['Gaspr.€/MWh']
        aTimeSeries = (datetime.datetime(2020, 1, 1) + np.arange(len(P_el_Last)) * datetime.timedelta(hours=0.25)).astype('datetime64')

        Strom, Fernwaerme, Gas, Kohle = cBus('el', 'Strom'), cBus('heat', 'Fernwärme'), cBus('fuel', 'Gas'), cBus('fuel', 'Kohle')
        costs, CO2, PE = cEffectType('costs', '€', 'Kosten', isStandard=True, isObjective=True), cEffectType('CO2', 'kg', 'CO2_e-Emissionen'), cEffectType('PE', 'kWh_PE', 'Primärenergie')

        aGaskessel = cKessel('Kessel', eta=0.85, Q_th=cFlow(label='Q_th', bus=Fernwaerme), Q_fu=cFlow(label='Q_fu', bus=Gas, nominal_val=95, min_rel=12 / 95, iCanSwitchOff=True, switchOnCosts=1000, valuesBeforeBegin=[0]))
        aKWK = cKWK('BHKW2', eta_th=0.58, eta_el=0.22, switchOnCosts=24000, P_el=cFlow('P_el', bus=Strom), Q_th=cFlow('Q_th', bus=Fernwaerme), Q_fu=cFlow('Q_fu', bus=Kohle, nominal_val=288, min_rel=87 / 288), on_valuesBeforeBegin=[0])
        aSpeicher = cStorage('Speicher', inFlow=cFlow('Q_th_load', nominal_val=137, bus=Fernwaerme), outFlow=cFlow('Q_th_unload', nominal_val=158, bus=Fernwaerme), capacity_inFlowHours=684, chargeState0_inFlowHours=137, charge_state_end_min=137, charge_state_end_max=158, eta_load=1, eta_unload=1, fracLossPerHour=0.001, avoidInAndOutAtOnce=True)

        TS_Q_th_Last, TS_P_el_Last = cTSraw(Q_th_Last), cTSraw(P_el_Last, agg_weight=0.7)
        aWaermeLast, aStromLast = cSink('Wärmelast', sink=cFlow('Q_th_Last', bus=Fernwaerme, nominal_val=1, val_rel=TS_Q_th_Last)), cSink('Stromlast', sink=cFlow('P_el_Last', bus=Strom, nominal_val=1, val_rel=TS_P_el_Last))
        aKohleTarif, aGasTarif = cSource('Kohletarif', source=cFlow('Q_Kohle', bus=Kohle, nominal_val=1000, costsPerFlowHour={costs: 4.6, CO2: 0.3})), cSource('Gastarif', source=cFlow('Q_Gas', bus=Gas, nominal_val=1000, costsPerFlowHour={costs: gP, CO2: 0.3}))

        p_feed_in, p_sell = cTSraw(-(p_el - 0.5), agg_type='p_el'), cTSraw(p_el + 0.5, agg_type='p_el')
        aStromEinspeisung, aStromTarif = cSink('Einspeisung', sink=cFlow('P_el', bus=Strom, nominal_val=1000, costsPerFlowHour=p_feed_in)), cSource('Stromtarif', source=cFlow('P_el', bus=Strom, nominal_val=1000, costsPerFlowHour={costs: p_sell, CO2: 0.3}))
        aStromEinspeisung.sink.costsPerFlowHour[None].setAggWeight(.5)
        aStromTarif.source.costsPerFlowHour[costs].setAggWeight(.5)

        es = cEnergySystem(aTimeSeries, dt_last=None)
        es.addEffects(costs, CO2, PE)
        es.addComponents(aGaskessel, aWaermeLast, aStromLast, aGasTarif, aKohleTarif, aStromEinspeisung, aStromTarif, aKWK, aSpeicher)

        if doFullCalc:
            calc = cCalculation('fullModel', es, 'pyomo')
            calc.doModelingAsOneSegment()
        if doSegmentedCalc:
            calc = cCalculation('segModel', es, 'pyomo')
            calc.doSegmentedModelingAndSolving(self.solverProps, segmentLen=97, nrOfUsedSteps=96)
        if doAggregatedCalc:
            calc = cCalculation('aggModel', es, 'pyomo')
            calc.doAggregatedModeling(6, 4, True, True, False, 0, 0, addPeakMax=[TS_Q_th_Last], addPeakMin=[TS_P_el_Last, TS_Q_th_Last])

        es.printModel()
        es.printVariables()
        es.printEquations()

        if not doSegmentedCalc:
            calc.solve(self.solverProps)

        import flixOpt.flixPostprocessing as flixPost
        return flixPost.flix_results(calc.nameOfCalc).results


if __name__ == '__main__':
    unittest.main()