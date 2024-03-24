# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:26:10 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

from flixOpt.flixStructure import *
from flixOpt.flixComps import *
from flixOpt.flixBasicsPublic import *
from typing import Literal, Union, List

def run_model(modeling_type: Literal["full", "segmented", "aggregated"]) -> Union[float, List]:
    # mögliche Testszenarien für testing-tool:
    # abschnittsweise linear testen
    # Komponenten mit offenen Flows
    # Binärvariablen ohne max-Wert-Vorgabe des Flows (Binärungenauigkeitsproblem)
    # Medien-zulässigkeit

    # solver:
    gapFrac = 0.0005
    solver_name = 'cbc'
    # solver_name    = 'gurobi'
    # solver_name    = 'glpk'
    solverProps = {'gapFrac': gapFrac, 'solver': solver_name, 'displaySolverOutput': True, 'threads': 16}

    nameSuffix = '_' + solver_name  # for saving-file

    ## Auswahl Rechentypen: ##
    doFullCalc = False
    doSegmentedCalc = False
    doAggregatedCalc = False
    if modeling_type == "full":
        doFullCalc = True
    elif modeling_type == "segmented":
        doSegmentedCalc = True
    elif modeling_type == "aggregated":
        doAggregatedCalc = True
    else:
        raise Exception("Unknown modeling type")

    ## segmented Properties: ##

    nrOfUsedSteps = 96 * 1
    segmentLen = nrOfUsedSteps + 1 * 96

    ## aggregated Properties: ##

    periodLengthInHours = 6
    noTypicalPeriods = 21
    noTypicalPeriods = 4
    useExtremeValues = True
    # useExtremeValues    = False
    fixBinaryVarsOnly = False
    # fixBinaryVarsOnly   = True
    fixStorageFlows = True
    # fixStorageFlows     = False
    percentageOfPeriodFreedom = 0
    costsOfPeriodFreedom = 0

    import pandas as pd
    import numpy as np
    import datetime
    import os
    import time

    calcFull = None
    calcSegs = None
    calcAgg = None

    # #########################################################################
    # ######################  Data Import  ####################################

    # Daten einlesen
    filename = os.path.join(os.path.dirname(__file__), "Zeitreihen2020.csv")
    ts_raw = pd.read_csv(filename, index_col=0)
    ts_raw = ts_raw.sort_index()

    # ts = ts_raw['2020-01-01 00:00:00':'2020-12-31 23:45:00']  # EDITIEREN FÜR ZEITRAUM
    ts = ts_raw['2020-01-01 00:00:00':'2020-12-31 23:45:00']
    # ts['Kohlepr.€/MWh'] = 4.6
    ts.set_index(pd.to_datetime(ts.index), inplace=True)  # str to datetime
    data = ts

    time_zero = time.time()

    # ENTWEDER...
    # Zeitaussschnitt definieren
    zeitraumInTagen = 366  # angeben!!!
    nrOfZeitschritte = zeitraumInTagen * 4 * 24
    data_sub = data[0:nrOfZeitschritte]

    # ODER....
    # data_sub = data['2020-01-01':'2020-01-07 23:45:00']
    data_sub = data['2020-01-01':'2020-01-01 23:45:00']
    data_sub = data['2020-07-01':'2020-07-07 23:45:00']
    # halbes Jahr:
    data_sub = data['2020-01-01':'2020-06-30 23:45:00']
    data_sub = data
    data_sub = data['2020-01-01':'2020-01-15 23:45:00']
    data_sub = data['2020-01-01':'2020-01-03 23:45:00']

    # Zeit-Index:
    aTimeIndex = data_sub.index
    aTimeIndex = aTimeIndex.to_pydatetime()  # datetime-Format

    ################ Bemerkungen: #################
    # jetzt hast du ein Dataframe
    # So kannst du ein Vektor aufrufen:
    P_el_Last = data_sub['P_Netz/MW']
    Q_th_Last = data_sub['Q_Netz/MW']
    p_el = data_sub['Strompr.€/MWh']

    HG_EK_min = 0
    HG_EK_max = 100000
    HG_VK_min = -100000
    HG_VK_max = 0
    gP = data_sub['Gaspr.€/MWh']

    #############################################################################
    nrOfPeriods = len(P_el_Last)
    # aTimeSeries = pd.date_range('1/1/2020',periods=nrOfPeriods,freq='15min')
    aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(nrOfPeriods) * datetime.timedelta(hours=0.25)
    aTimeSeries = aTimeSeries.astype('datetime64')

    ##########################################################################

    import pandas as pd
    import logging as log
    import os  # für logging

    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    log.warning('test warning')
    log.info('test info')
    log.debug('test debung')

    print('#######################################################################')
    print('################### start of modeling #################################')

    # Bus-Definition:
    #                 Typ         Name
    excessCosts = 1e5
    excessCosts = None
    Strom = cBus('el', 'Strom', excessCostsPerFlowHour=excessCosts);
    Fernwaerme = cBus('heat', 'Fernwärme', excessCostsPerFlowHour=excessCosts);
    Gas = cBus('fuel', 'Gas', excessCostsPerFlowHour=excessCosts);
    Kohle = cBus('fuel', 'Kohle', excessCostsPerFlowHour=excessCosts);

    # Effects

    costs = cEffectType('costs', '€', 'Kosten', isStandard=True, isObjective=True)
    CO2 = cEffectType('CO2', 'kg', 'CO2_e-Emissionen')  # effectsPerFlowHour = {'costs' : 180} ))
    PE = cEffectType('PE', 'kWh_PE', 'Primärenergie')

    # Komponentendefinition:

    aGaskessel = cKessel('Kessel', eta=0.85,  # , costsPerRunningHour = {costs:0,CO2:1000},#, switchOnCosts = 0
                         Q_th=cFlow(label='Q_th', bus=Fernwaerme),  # maxGradient = 5),
                         Q_fu=cFlow(label='Q_fu', bus=Gas, nominal_val=95, min_rel=12 / 95, iCanSwitchOff=True,
                                    switchOnCosts=1000, valuesBeforeBegin=[0]))

    aKWK = cKWK('BHKW2', eta_th=0.58, eta_el=0.22, switchOnCosts=24000,
                P_el=cFlow('P_el', bus=Strom),
                Q_th=cFlow('Q_th', bus=Fernwaerme),
                Q_fu=cFlow('Q_fu', bus=Kohle, nominal_val=288, min_rel=87 / 288), on_valuesBeforeBegin=[0])

    aSpeicher = cStorage('Speicher',
                         inFlow=cFlow('Q_th_load', nominal_val=137, bus=Fernwaerme),
                         outFlow=cFlow('Q_th_unload', nominal_val=158, bus=Fernwaerme),
                         capacity_inFlowHours=684,
                         chargeState0_inFlowHours=137,
                         charge_state_end_min=137,
                         charge_state_end_max=158,
                         eta_load=1, eta_unload=1,
                         fracLossPerHour=0.001,
                         avoidInAndOutAtOnce=True)

    TS_Q_th_Last = cTSraw(Q_th_Last)
    aWaermeLast = cSink('Wärmelast', sink=cFlow('Q_th_Last', bus=Fernwaerme, nominal_val=1, val_rel=TS_Q_th_Last))

    # TS with explicit defined weight
    TS_P_el_Last = cTSraw(P_el_Last, agg_weight=0.7)  # explicit defined weight
    aStromLast = cSink('Stromlast', sink=cFlow('P_el_Last', bus=Strom, nominal_val=1, val_rel=TS_P_el_Last))

    aKohleTarif = cSource('Kohletarif',
                          source=cFlow('Q_Kohle', bus=Kohle, nominal_val=1000, costsPerFlowHour={costs: 4.6, CO2: 0.3}))

    aGasTarif = cSource('Gastarif',
                        source=cFlow('Q_Gas', bus=Gas, nominal_val=1000, costsPerFlowHour={costs: gP, CO2: 0.3}))

    # 2 TS with same aggType (--> implicit defined weigth = 0.5)
    p_feed_in = cTSraw(-(p_el - 0.5), agg_type='p_el')  # weight shared in group p_el
    p_sell = cTSraw(p_el + 0.5, agg_type='p_el')
    # p_feed_in = p_feed_in.value # only value
    # p_sell    = p_sell.value # only value
    aStromEinspeisung = cSink('Einspeisung', sink=cFlow('P_el', bus=Strom, nominal_val=1000, costsPerFlowHour=p_feed_in))
    aStromEinspeisung.sink.costsPerFlowHour[None].setAggWeight(.5)

    aStromTarif = cSource('Stromtarif',
                          source=cFlow('P_el', bus=Strom, nominal_val=1000, costsPerFlowHour={costs: p_sell, CO2: 0.3}))
    aStromTarif.source.costsPerFlowHour[costs].setAggWeight(.5)

    # Zusammenführung:
    es = cEnergySystem(aTimeSeries, dt_last=None)
    # es.addComponents(aGaskessel,aWaermeLast,aGasTarif)#,aGaskessel2)
    es.addEffects(costs)
    es.addEffects(CO2, PE)
    es.addComponents(aGaskessel, aWaermeLast, aStromLast, aGasTarif, aKohleTarif)
    es.addComponents(aStromEinspeisung, aStromTarif)
    es.addComponents(aKWK)

    es.addComponents(aSpeicher)

    # es.mainSystem.extractSubSystem([0,1,2])


    chosenEsTimeIndexe = None
    # chosenEsTimeIndexe = [1,3,5]

    ########################
    ######## Lösung ########
    listOfCalcs = []

    # Roh-Rechnung:
    if doFullCalc:
        calcFull = cCalculation('fullModel', es, 'pyomo', chosenEsTimeIndexe)
        calcFull.doModelingAsOneSegment()

        es.printModel()
        es.printVariables()
        es.printEquations()

        calcFull.solve(solverProps, nameSuffix=nameSuffix)
        listOfCalcs.append(calcFull)

    # segmentierte Rechnung:
    if doSegmentedCalc:
        calcSegs = cCalculation('segModel', es, 'pyomo', chosenEsTimeIndexe)
        calcSegs.doSegmentedModelingAndSolving(solverProps, segmentLen=segmentLen, nrOfUsedSteps=nrOfUsedSteps,
                                               nameSuffix=nameSuffix)
        listOfCalcs.append(calcSegs)

    # aggregierte Berechnung:

    if doAggregatedCalc:
        calcAgg = cCalculation('aggModel', es, 'pyomo')
        calcAgg.doAggregatedModeling(periodLengthInHours,
                                     noTypicalPeriods,
                                     useExtremeValues,
                                     fixStorageFlows,
                                     fixBinaryVarsOnly,
                                     percentageOfPeriodFreedom=percentageOfPeriodFreedom,
                                     costsOfPeriodFreedom=costsOfPeriodFreedom,
                                     addPeakMax=[TS_Q_th_Last],  # add timeseries of period with maxPeak explicitly
                                     addPeakMin=[TS_P_el_Last, TS_Q_th_Last]
                                     )

        es.printVariables()
        es.printEquations()

        calcAgg.solve(solverProps, nameSuffix=nameSuffix)
        listOfCalcs.append(calcAgg)

    ####### loading #######

    import flixOpt.flixPostprocessing as flixPost

    listOfResults = []

    if doFullCalc:
        full = flixPost.flix_results(calcFull.nameOfCalc)
        listOfResults.append(full)
        # del calcFull

        costs = full.results_struct.globalComp.costs.all.sum

    if doAggregatedCalc:
        agg = flixPost.flix_results(calcAgg.nameOfCalc)
        listOfResults.append(agg)
        # del calcAgg
        costs = agg.results_struct.globalComp.costs.all.sum

    if doSegmentedCalc:
        seg = flixPost.flix_results(calcSegs.nameOfCalc)
        listOfResults.append(seg)
        # del calcSegs
        costs = seg.results_struct.globalComp.costs.all.sum

    return costs