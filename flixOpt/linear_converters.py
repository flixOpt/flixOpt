# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
import logging

from flixOpt import Flow, utils
from flixOpt.components import LinearConverter
from flixOpt.core import Numeric_TS

logger = logging.getLogger('flixOpt')


class Boiler(LinearConverter):
    def __init__(self,
                 label: str,
                 eta: Numeric_TS,
                 Q_fu: Flow,
                 Q_th: Flow):
        """
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
        """
        super().__init__(label, inputs=[Q_fu], outputs=[Q_th], conversion_factors=[{Q_fu: eta, Q_th: 1}])

        self.eta = eta
        self.Q_fu = Q_fu
        self.Q_th = Q_th

        utils.check_bounds(eta, 'eta', 0+1e-10, 1-1e-10)


class Power2Heat(LinearConverter):
    def __init__(self,
                 label: str,
                 eta: Numeric_TS,
                 P_el: Flow,
                 Q_th: Flow,
                 **kwargs):
        """
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

        """
        super().__init__(label, inputs=[P_el], outputs=[Q_th], conversion_factors=[{P_el: eta, Q_th: 1}], **kwargs)

        self.eta = eta
        self.P_el = P_el
        self.Q_th = Q_th

        utils.check_bounds(eta, 'eta',                0+1e-10, 1-1e-10)


class HeatPump(LinearConverter):
    def __init__(self,
                 label: str,
                 COP: Numeric_TS,
                 P_el: Flow,
                 Q_th: Flow,
                 **kwargs):
        """
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
        """
        super().__init__(label, inputs=[P_el], outputs=[Q_th], conversion_factors=[{P_el: COP, Q_th: 1}], **kwargs)

        self.COP = COP
        self.P_el = P_el
        self.Q_th = Q_th

        utils.check_bounds(COP, 'COP',  1 + 1e-10, 20 - 1e-10)


class CoolingTower(LinearConverter):
    def __init__(self,
                 label: str,
                 specific_electricity_demand: Numeric_TS,
                 P_el:Flow,
                 Q_th:Flow,
                 **kwargs):
        """
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

        """
        super().__init__(label, inputs=[P_el, Q_th],
                         conversion_factors=[{P_el: 1, Q_th: -specific_electricity_demand}], **kwargs)

        self.specificElectricityDemand = specific_electricity_demand
        self.P_el = P_el
        self.Q_th = Q_th

        utils.check_bounds(specific_electricity_demand, 'specific_electricity_demand', 0, 1)


class CHP(LinearConverter):
    def __init__(self,
                 label: str,
                 eta_th: Numeric_TS,
                 eta_el: Numeric_TS,
                 Q_fu: Flow,
                 P_el: Flow,
                 Q_th: Flow,
                 **kwargs):
        """
        constructor of cCHP

        Parameters
        ----------
        label : str
            name of CHP-unit.
        eta_th : float or TS
            thermal efficiency.
        eta_el : float or TS
            electrical efficiency.
        Q_fu : cFlow
            fuel input-flow.
        P_el : cFlow
            electricity output-flow.
        Q_th : cFlow
            heat output-flow.
        **kwargs :

        """
        heat = {Q_fu: eta_th, Q_th: 1}
        electricity = {Q_fu: eta_el, P_el: 1}

        super().__init__(label, inputs=[Q_fu], outputs=[Q_th, P_el], conversion_factors=[heat, electricity], **kwargs)

        # args to attributes:
        self.eta_th = eta_th
        self.eta_el = eta_el
        self.Q_fu = Q_fu
        self.P_el = P_el
        self.Q_th = Q_th

        utils.check_bounds(eta_th, 'eta_th',                0+1e-10, 1-1e-10)
        utils.check_bounds(eta_el, 'eta_el',                0+1e-10, 1-1e-10)
        utils.check_bounds(eta_el+eta_th, 'eta_th+eta_el',  0+1e-10, 1-1e-10)


class HeatPumpWithSource(LinearConverter):
    def __init__(self,
                 label: str,
                 COP: Numeric_TS,
                 P_el: Flow,
                 Q_ab: Flow,
                 Q_th: Flow,
                 **kwargs):
        """
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
        """

        # super:
        electricity = {P_el: COP, Q_th: 1}
        heat_source = {Q_ab: COP / (COP - 1), Q_th: 1}

        super().__init__(label, inputs=[P_el, Q_ab], outputs=[Q_th],
                         conversion_factors=[electricity, heat_source], **kwargs)

        self.COP = COP
        self.P_el = P_el
        self.Q_ab = Q_ab
        self.Q_th = Q_th

        utils.check_bounds(COP, 'eta_th', 0 + 1e-10, 20 - 1e-10)
