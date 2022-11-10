# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:23:25 2022

@author: Panitz
"""

# just for testing values - not up to date!

import Model_and_solve

if Model_and_solve.doFullCalc:
    costs = Model_and_solve.full.results_struct.globalComp.costs.all.sum
    if (np.round(costs) == np.round(361984.0791699171)):
        print('################')
        print('full: test is ok') 
    else: 
        raise Exception('test is not ok')
if Model_and_solve.doAggregatedCalc:
    costs = Model_and_solve.agg.results_struct.globalComp.costs.all.sum
    if (np.round(costs) == np.round(359613.8834453795)):
        print('################')
        print('agg: test is ok')
    else:
        raise Exception('test is not ok')

if Model_and_solve.doSegmentedCalc:
    costs = Model_and_solve.seg.results_struct.globalComp.costs.all.sum
    if all (np.round(costs) == np.round([210654.98636908, 269804.58925733])):
        print('################')
        print('seg: test is ok')    
    else:
        raise Exception('test is not ok')        
