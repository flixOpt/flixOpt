# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

from .flixStructure import *
from .flixFeatures import *
from .flixComps import LinearTransformer, cKWK
from flixOpt.flixOptHelperFcts import checkExists


def KWKektA(label: str, nominal_val: float, BusFuel: cBus, BusTh: cBus, BusEl: cBus,
            eta_th: list, eta_el: list, exists=None, group=None, **kwargs) -> list:
    '''
    EKT A - Modulation, linear interpolation

    Creates a KWK with a variable rate between electricity and heat production.
    Properties:
        Modulation of Total Power (Fuel) [min_rel, max_rel, nominal value]
        linear interpolation between efficiencies A and B

        Not working: InvestParameters with variable Size

    Use in Following manner:
        #  KWK_poweroriented    KWK_heatoriented,
    eta_th =   [0.00001,            0.9 ]
    eta_el =   [0.2,                0.1 ]
    Q_th =   [ cFlow(),            cFlow() ]
    P_el =   [ cFlow(),            cFlow() ]

    Parameters
    ----------
    label : str
        name of CHP-unit.
    eta_th : list of float or array
        thermal efficiency of poweroriented (po) and heatoriented (ho) operation
        passed in a list [eta_th_po, eta_th_ho]. the elements can be float or array
    eta_el : list of float or array
        electrical efficiency of poweroriented (po) and heatoriented (ho) operation.
        passed in a list [eta_th_po, eta_th_ho]. the elements can be float or array
    Q_fu : cFlow
        fuel input-flow.
    P_el : list of cFlow
        electricity output-flow.
    Q_th : list of cFlow
        heat output-flow.
    **kwargs :
        Additional keyword arguments. Passed to the fule input flow!!

    Returns
    -------
    list(LinearTransformer, cKWK, cKWK)
            a list of Components that need to be added to the cEnergySystem
    '''

    # filtering,because eta can not be 0
    for eta in [eta_el, eta_th]:
        for i in range(len(eta)):
            if isinstance(eta[i], (float, int)) and eta[i] == 0:
                eta[i] = 0.0000001
            elif isinstance(eta[i], (np.ndarray)):
                eta[i][eta[i] == 0] = 0.0000001

    eta_thA = eta_th[0]
    eta_thB = eta_th[1]
    eta_elB = eta_el[0]
    eta_elA = eta_el[1]

    HelperBus = cBus(label='Helper' + label + 'In', media=None)  # balancing node/bus of electricity

    # Transformer 1
    Qin = cFlow(label="Qfu", bus=BusFuel, nominal_val=nominal_val, min_rel=1, **kwargs)
    Qout = cFlow(label="Helper" + label + 'Fu', bus=HelperBus)
    EKTIn = LinearTransformer(label=label + "In", exists=exists, group=group,
                              inputs=[Qin], outputs=[Qout], factor_Sets=[{Qin: 1, Qout: 1}])
    # EKT A
    EKTA = cKWK(label=label + "A", exists=exists, group=group,
                eta_th=eta_thA, eta_el=eta_elA,
                P_el=cFlow(label="Pel", bus=BusEl),
                Q_fu=cFlow(label="Helper" + label + 'A', bus=HelperBus),
                Q_th=cFlow(label="Qth", bus=BusTh))
    # EKT B
    EKTB = cKWK(label=label + "B", exists=exists, group=group,
                eta_th=eta_thB, eta_el=eta_elB,
                P_el=cFlow(label="Pel", bus=BusEl),
                Q_fu=cFlow(label="Helper" + label + 'B', bus=HelperBus),
                Q_th=cFlow(label="Qth", bus=BusTh))
    return [EKTIn, EKTA, EKTB]


def KWKektB(label: str, BusFuel: cBus, BusTh: cBus, BusEl: cBus,
            nominal_val_Qfu: float, segQth: list[float], segPel: list[float],
            costsPerFlowHour_fuel: dict = None, costsPerFlowHour_th: dict = None, costsPerFlowHour_el: dict = None,
            iCanSwitchOff=True, exists=1, group=None, invest_parameters: InvestParameters = None, **kwargs) -> list:
    '''
    EKT B - On/Off, interpolation with Base Points
    Creates a KWK with a variable rate between electricity and heat production

    Properties:
        On/Off-operation
        Interpolation with Base Points between efficiencies A and B

        Not working:
        InvestParameters with variable Size
        Variation of total Power

    Nominal Value is equal to the max of seqFu

    Parameters
    ----------
    label: str
        A string representing the label for the component.
    BusFuel: cBus
        The bus representing the fuel input for the component.
    BusTh: cBus
        The bus representing the thermal output for the component.
    BusEl: cBus
        The bus representing the electrical output for the component.
    nominal_val_Qfu: float
        Fuel flow. Constant, But iCanSwitchOff=True
    segQth: list[float]
        Expression with Base Points for thermal flow.
        [2, 5, 9]
    segPel: list[float]
        Expression with Base Points for electrical power.
        [3, 1, 0]
    costsPerFlowHour_fuel: dict
        A dictionary representing the costs associated with fuel input. cEffect as keys
    costsPerFlowHour_el: dict
        A dictionary representing the costs associated with electricity output. cEffect as keys
    costsPerFlowHour_th: dict
        A dictionary representing the costs associated with thermal output. cEffect as keys
    iCanSwitchOff: bool, optional
        Wether the Component can be switched off. Default True
    exists: any, optional
        A parameter specifying when the component exists. Defaults to 1.
    group: any, optional
        A parameter specifying the group to which the component belongs. Defaults to None.
    invest_parameters: InvestParameters, optional
        An object containing investment-related parameters. Defaults to None. Passed to the thermal output flow
    **kwargs
        Additional keyword arguments. Passed to the input fuel flow. Allowed keywords see documentation of cFlow

    Returns
    -------
    list(LinearTransformer, LinearTransformer, LinearTransformer)
        a list of Components that need to be added to the cEnergySystem

    Raises
    ------
    Exception
        Raised if minmax_rel is not 0 or 1, or if minmax_rel contains values other than 0 and 1.
    '''

    # Create Lists for segments_of_flows
    segQfu_el = np.linspace(start=0.0001, stop=nominal_val_Qfu, num=len(segPel)).tolist()
    segQfu_th = segQfu_el[::-1]  # reversed
    # Apply proper formating for segments of flows, rounding to avoid numerical error, which leads to excess in HelperBus
    # TODO: Is this necessary?
    nominal_val_Qfu = round(nominal_val_Qfu, 4)
    segQfu_el = [num for num in segQfu_el for _ in range(2)][1:-1]
    segQfu_th = [num for num in segQfu_th for _ in range(2)][1:-1]
    segQth = [num for num in segQth for _ in range(2)][1:-1]
    segPel = [num for num in segPel for _ in range(2)][1:-1]

    segQfu_el = [round(num, 4) for num in segQfu_el]
    segQfu_th = [round(num, 4) for num in segQfu_th]
    segQth = [round(num, 4) for num in segQth]
    segPel = [round(num, 4) for num in segPel]

    if iCanSwitchOff:
        segQfu_el = [0, 0] + segQfu_el
        segQfu_th = [0, 0] + segQfu_th
        segQth = [0, 0] + segQth
        segPel = [0, 0] + segPel

    HelperBus = cBus(label='Helper' + label + 'In', media=None,
                     excessCostsPerFlowHour=None)  # balancing node/bus of electricity
    # Handling min_rel and max_rel
    max_rel = kwargs.pop("max_rel", 1)
    checkExists(max_rel)

    # Transformer 1
    Qin = cFlow(label="Qfu", bus=BusFuel, nominal_val=nominal_val_Qfu, min_rel=max_rel, max_rel=max_rel,
                costsPerFlowHour=costsPerFlowHour_fuel, **kwargs)
    Qout = cFlow(label="Helper" + label + 'Fu', bus=HelperBus)
    EKTIn = LinearTransformer(label=label + "In", exists=exists, group=group,
                              inputs=[Qin], outputs=[Qout], factor_Sets=[{Qin: 1, Qout: 1}])

    # Transformer Strom
    P_el = cFlow(label="Pel", bus=BusEl, nominal_val=max(segPel), costsPerFlowHour=costsPerFlowHour_el)
    Q_fu = cFlow(label="Helper" + label + 'A', bus=HelperBus, nominal_val=nominal_val_Qfu)
    segs_el = {Q_fu: segQfu_el, P_el: segPel.copy()}
    EKTA = LinearTransformer(label=label + "A", exists=exists, group=group,
                             outputs=[P_el], inputs=[Q_fu], segmentsOfFlows=segs_el)

    # Transformer Wärme
    Q_th = cFlow(label="Qth", bus=BusTh, nominal_val=max(segQth), costsPerFlowHour=costsPerFlowHour_th,
                 invest_parameters=invest_parameters)
    Q_fu2 = cFlow(label="Helper" + label + 'B', bus=HelperBus)
    segments = {Q_fu2: segQfu_th, Q_th: segQth}
    EKTB = LinearTransformer(label=label + "B", exists=exists, group=group,
                             outputs=[Q_th], inputs=[Q_fu2], segmentsOfFlows=segments)

    return [EKTIn, EKTA, EKTB]
