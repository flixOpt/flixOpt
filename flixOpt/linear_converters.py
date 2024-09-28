# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
from flixOpt import Flow, utils, TimeSeriesRaw
from flixOpt.components import LinearConverter
from flixOpt.core import Numeric_TS, TimeSeries
from flixOpt.elements import MediumCollection


class Boiler(LinearConverter):
    """
    class Boiler
    """
    new_init_args = ['label', 'eta', 'Q_fu', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'conversion_factors']

    def __init__(self, label: str, eta: Numeric_TS, Q_fu: Flow, Q_th: Flow, **kwargs):
        '''
        constructor for boiler

        Parameters
        ----------
        label : str
            name of bolier.
        eta : float or TS
            thermal efficiency.
        Q_fu : Flow
            fuel input-flow
        Q_th : Flow
            thermal output-flow.
        **kwargs : see mother classes!


        '''
        # super:
        kessel_bilanz = {Q_fu: eta,
                         Q_th: 1}  # eq: eta * Q_fu = 1 * Q_th # TODO: Achtung eta ist hier noch nicht TS-vector!!!

        super().__init__(label, inputs=[Q_fu], outputs=[Q_th], conversion_factors=[kessel_bilanz], **kwargs)

        # invest_parameters to attributes:
        self.eta = TimeSeries('eta', eta, self)  # thermischer Wirkungsgrad
        self.Q_fu = Q_fu
        self.Q_th = Q_th

        # allowed medium:
        Q_fu.set_medium_if_not_set(MediumCollection.fuel)
        Q_th.set_medium_if_not_set(MediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_th < 1
        utils.check_bounds(eta, 'eta', self.eta_bounds[0], self.eta_bounds[1])

        # # generische property für jeden Koeffizienten
        # self.eta = property(lambda s: s.__get_coeff('eta'), lambda s,v: s.__set_coeff(v,'eta'))


class Power2Heat(LinearConverter):
    """
    class Power2Heat
    """
    new_init_args = ['label', 'eta', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'conversion_factors']

    def __init__(self, label:str, eta:Numeric_TS, P_el:Flow, Q_th:Flow, **kwargs):
        '''
        constructor for boiler

        Parameters
        ----------
        label : str
            name of bolier.
        eta : float or TS
            thermal efficiency.
        P_el : Flow
            electric input-flow
        Q_th : Flow
            thermal output-flow.
        **kwargs : see mother classes!


        '''
        # super:
        kessel_bilanz = {P_el: eta,
                         Q_th: 1}  # eq: eta * Q_fu = 1 * Q_th # TODO: Achtung eta ist hier noch nicht TS-vector!!!

        super().__init__(label, inputs=[P_el], outputs=[Q_th], conversion_factors=[kessel_bilanz], **kwargs)

        # invest_parameters to attributes:
        self.eta = TimeSeries('eta', eta, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.set_medium_if_not_set(MediumCollection.el)
        Q_th.set_medium_if_not_set(MediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_th < 1
        utils.check_bounds(eta, 'eta', self.eta_bounds[0], self.eta_bounds[1])

        # # generische property für jeden Koeffizienten
        # self.eta = property(lambda s: s.__get_coeff('eta'), lambda s,v: s.__set_coeff(v,'eta'))


class HeatPump(LinearConverter):
    """
    class HeatPump
    """
    new_init_args = ['label', 'COP', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'conversion_factors']

    def __init__(self, label:str, COP:Numeric_TS, P_el:Flow, Q_th:Flow, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of heatpump.
        COP : float or TS
            Coefficient of performance.
        P_el : Flow
            electricity input-flow.
        Q_th : Flow
            thermal output-flow.
        **kwargs : see motherclasses
        '''

        # super:
        heatPump_bilanz = {P_el: COP, Q_th: 1}  # TODO: Achtung eta ist hier noch nicht TS-vector!!!
        super().__init__(label, inputs=[P_el], outputs=[Q_th], conversion_factors=[heatPump_bilanz], **kwargs)

        # invest_parameters to attributes:
        self.COP = TimeSeries('COP', COP, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.set_medium_if_not_set(MediumCollection.el)
        Q_th.set_medium_if_not_set(MediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 20 - 1e-10]  # 0 < COP < 1
        utils.check_bounds(COP, 'COP', self.eta_bounds[0], self.eta_bounds[1])


class CoolingTower(LinearConverter):
    """
    Klasse CoolingTower
    """
    new_init_args = ['label', 'specificElectricityDemand', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'conversion_factors']

    def __init__(self, label:str, specificElectricityDemand:Numeric_TS, P_el:Flow, Q_th:Flow, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of cooling tower.
        specificElectricityDemand : float or TS
            auxiliary electricty demand per cooling power, i.g. 0.02 (2 %).
        P_el : Flow
            electricity input-flow.
        Q_th : Flow
            thermal input-flow.
        **kwargs : see getKwargs() and their description in motherclasses

        '''
        # super:
        auxElectricity_eq = {P_el: 1,
                             Q_th: -specificElectricityDemand}  # eq: 1 * P_el - specificElectricityDemand * Q_th = 0  # TODO: Achtung eta ist hier noch nicht TS-vector!!!
        super().__init__(label, inputs=[P_el, Q_th], outputs=[], conversion_factors=[auxElectricity_eq], **kwargs)

        # invest_parameters to attributes:
        self.specificElectricityDemand = TimeSeries('specificElectricityDemand', specificElectricityDemand,
                                                    self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.set_medium_if_not_set(MediumCollection.el)
        Q_th.set_medium_if_not_set(MediumCollection.heat)

        # Plausibilität eta:
        self.specificElectricityDemand_bounds = [0, 1]  # 0 < eta_th < 1
        utils.check_bounds(specificElectricityDemand, 'specificElectricityDemand',
                             self.specificElectricityDemand_bounds[0], self.specificElectricityDemand_bounds[1])


class CHP(LinearConverter):
    """
    class of combined heat and power unit (CHP)
    """
    new_init_args = ['label', 'eta_th', 'eta_el', 'Q_fu', 'P_el', 'Q_th']
    not_used_args = ['label', 'inputs', 'outputs', 'conversion_factors']

    # eta = 1 # Thermischer Wirkungsgrad
    # __eta_bound = [0,1]

    def __init__(self, label:str, eta_th:Numeric_TS, eta_el:Numeric_TS, Q_fu:Flow, P_el:Flow, Q_th:Flow, **kwargs):
        '''
        constructor of cCHP

        Parameters
        ----------
        label : str
            name of CHP-unit.
        eta_th : float or TS
            thermal efficiency.
        eta_el : float or TS
            electrical efficiency.
        Q_fu : Flow
            fuel input-flow.
        P_el : Flow
            electricity output-flow.
        Q_th : Flow
            heat output-flow.
        **kwargs :

        '''

        # super:
        waerme_glg = {Q_fu: eta_th, Q_th: 1}
        strom_glg = {Q_fu: eta_el, P_el: 1}
        #                      inputs         outputs               lineare Gleichungen
        super().__init__(label, inputs=[Q_fu], outputs=[P_el, Q_th], conversion_factors=[waerme_glg, strom_glg], **kwargs)

        # invest_parameters to attributes:
        self.eta_th = TimeSeries('eta_th', eta_th, self)
        self.eta_el = TimeSeries('eta_el', eta_el, self)
        self.Q_fu = Q_fu
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        Q_fu.set_medium_if_not_set(MediumCollection.fuel)
        Q_th.set_medium_if_not_set(MediumCollection.heat)
        P_el.set_medium_if_not_set(MediumCollection.el)

        # Plausibilität eta:
        self.eta_th_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_th < 1
        self.eta_el_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_el < 1

        utils.check_bounds(eta_th, 'eta_th', self.eta_th_bounds[0], self.eta_th_bounds[1])
        utils.check_bounds(eta_el, 'eta_el', self.eta_el_bounds[0], self.eta_el_bounds[1])
        utils.check_bounds(eta_th + eta_el, 'eta_th+eta_el',
                             self.eta_th_bounds[0]+self.eta_el_bounds[0],
                             self.eta_th_bounds[1]+self.eta_el_bounds[1])


class HeatPumpWithSource(LinearConverter):
    """
    class HeatPumpWithSource
    """
    new_init_args = ['label', 'COP', 'Q_ab', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'conversion_factors']

    def __init__(self, label:str, COP:Numeric_TS, P_el:Flow, Q_ab:Flow, Q_th:Flow, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of heatpump.
        COP : float, TS
            Coefficient of performance.
        Q_ab : Flow
            Heatsource input-flow.
        P_el : Flow
            electricity input-flow.
        Q_th : Flow
            thermal output-flow.
        **kwargs : see motherclasses
        '''

        # super:
        heatPump_bilanzEl = {P_el: COP, Q_th: 1}
        if isinstance(COP, TimeSeriesRaw):
            COP = COP.value
            heatPump_bilanzAb = {Q_ab: COP / (COP - 1), Q_th: 1}
        else:
            heatPump_bilanzAb = {Q_ab: COP / (COP - 1), Q_th: 1}
        super().__init__(label, inputs=[P_el, Q_ab], outputs=[Q_th],
                         conversion_factors=[heatPump_bilanzEl, heatPump_bilanzAb], **kwargs)

        # invest_parameters to attributes:
        self.COP = TimeSeries('COP', COP, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_ab = Q_ab
        self.Q_th = Q_th

        # allowed medium:
        P_el.set_medium_if_not_set(MediumCollection.el)
        Q_th.set_medium_if_not_set(MediumCollection.heat)
        Q_ab.set_medium_if_not_set(MediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 20 - 1e-10]  # 0 < COP < 1
        utils.check_bounds(COP, 'COP', self.eta_bounds[0], self.eta_bounds[1])
