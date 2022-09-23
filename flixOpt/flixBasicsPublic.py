# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:25:43 2022

@author: Panitz
"""

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
    
    def __init__(self, value, agg_type = None, agg_weight= None):
        self.value = value
        self.agg_type = agg_type
        self.agg_weight = agg_weight        
        if (agg_type is not None) and (agg_weight is not None):
            raise Exception('Either <agg_type> or explicit <agg_weigth> can be set. Not both!')
    
    