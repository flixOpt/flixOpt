# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:25:43 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
import pprint

# Anmerkung: cTSraw separat von cTS_vector wg. Einfachheit für Anwender
class cTSraw:
    '''
    timeseries class for transmit timeseries AND special characteristics of timeseries, 
    i.g. to define weights needed in calcType 'aggregated'
        EXAMPLE solar:
        you have several solar timeseries. These should not be overweighted 
        compared to the remaining timeseries (i.g. heat load, price)!
        val_rel_solar1 = cTS(sol_array_1, type = 'solar')
        val_rel_solar2 = cTS(sol_array_2, type = 'solar')
        val_rel_solar3 = cTS(sol_array_3, type = 'solar')    
        --> this 3 series of same type share one weight, i.e. internally assigned each weight = 1/3 
        (instead of standard weight = 1)
        
    Parameters
    ----------
    value: 
        scalar, array, np.array.
    agg_weight: 
        weight for calcType 'aggregated'; between 0..1, normally 1.    
    '''

    def __init__(self, value, agg_type=None, agg_weight=None):
        self.value = value
        self.agg_type = agg_type
        self.agg_weight = agg_weight
        if (agg_type is not None) and (agg_weight is not None):
            raise Exception('Either <agg_type> or explicit <agg_weigth> can be set. Not both!')

    def __repr__(self):
        return f"<cTSraw agg_type={self.agg_type!r}, agg_weight={self.agg_weight!r}>"

    def __str__(self):
        agg_info = f"agg_type={self.agg_type}, agg_weight={self.agg_weight}" if self.agg_type or self.agg_weight else "no aggregation info"
        return f"Timeseries: {agg_info}"


# Sammlung von Props für Investitionskosten (für cFeatureInvest)
class cInvestArgs:
    '''
    collects arguments for invest-stuff
    '''

    def __init__(self,
                 fixCosts=None,
                 divestCosts=None,
                 investmentSize_is_fixed=True,
                 investment_is_optional=True,  # Investition ist weglassbar
                 specificCosts=0,  # costs per Flow-Unit/Storage-Size/...
                 costsInInvestsizeSegments=None,
                 min_investmentSize=0,  # nur wenn nominal_val_is_fixed = False
                 max_investmentSize=1e9,  # nur wenn nominal_val_is_fixed = False
                 **kwargs):
        '''
        Parameters
        ----------
        fixCosts : None ore scalar
            fixed investment-costs if invested             
            (Attention: Annualize costs to chosen period!)
        divestCosts : None or scalar 
            fixed divestment-costs (if not invested, i.g. demolition costs or contractual penalty)
        investmentSize_is_fixed: boolean
            # True: fixed nominal_value; false: nominal_value as optimization-variable
        investment_is_optional: boolean
            if True: investment is not forced.
        specificCosts: scalar or cost-dict, i.g. {costs: 3, CO2: 0.3}      
            specific costs, i.g. in €/kW_nominal or €/m²_nominal    
            (Attention: Annualize costs to chosen period!)
        costsInInvestsizeSegments: 
            linear relation in segments, [invest_segments, cost_segments]
            with this you can also realise valid segments of investSize (and gaps of non-available sizes)
            
            example 1:
                
            >>> [[5,25,25,100], # nominal_value in kW
            >>>  {costs:[50,250,250,800], # €
            >>>   PE:[5,25,25,100]} # kWh_PrimaryEnergy
            >>> ]
            
            example 2 (if only standard-effect):
                
            >>> [[5,25,25,100], #kW # nominal_value in kW
            >>>  [50,250,250,800],#€ 
            >>> ]                
            
            (Attention: Annualize costs to chosen period!)
            (args 'specificCosts' and 'fixCosts' can be used in parallel to InvestsizeSegments)
        
        min_investmentSize: scalar
            Min nominal value (only if: nominal_val_is_fixed = False)
        max_investmentSize: scalar
            Max nominal value (only if: nominal_val_is_fixed = False)
  
        '''

        self.fixCosts = fixCosts
        self.divestCosts = divestCosts
        self.investmentSize_is_fixed = investmentSize_is_fixed
        self.investment_is_optional = investment_is_optional
        self.specificCosts = specificCosts
        self.costsInInvestsizeSegments = costsInInvestsizeSegments
        self.min_investmentSize = min_investmentSize
        self.max_investmentSize = max_investmentSize

        super().__init__(**kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.__dict__}"

    def __str__(self):
        details = [
            f"fixCosts={self.fixCosts}" if self.fixCosts else ""
            f"divestCosts={self.divestCosts}" if self.divestCosts else ""
            f"specificCosts={self.specificCosts}" if self.specificCosts else ""
            f"Fixed Size" if self.investmentSize_is_fixed else ""
            f"Optional" if self.investment_is_optional else ""
            f"min/max_Size=[{self.min_investmentSize}-{self.max_investmentSize}]"
            f"costsInInvestsizeSegments={self.costsInInvestsizeSegments}, " if self.costsInInvestsizeSegments else ""
        ]

        all_relevant_parts = [part for part in details if part != ""]

        full_str =f"{', '.join(all_relevant_parts)}"

        return f"<{self.__class__.__name__}>: {full_str}"


