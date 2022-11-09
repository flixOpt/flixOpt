# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:54:19 2022

@author: Panitz
"""

import example_complex

calc1= example_complex.calc1
# ## TESTING: ##
print('#############')
print('## Testing ##')
objective_value = calc1.infos['modboxes']['info'][0]['main_results']['Result of Obj']
assert round(objective_value, -1) == -11600, '!!!Achtung Ergebnis-Änderung!!!'
print('##   ok!   ##')
print('#############')

# if nameOfCalcSegs is not None:
#     assert round(0.001* sum(calcSegs.results_struct.globalComp.costs.operation.sum_TS )) == -12, 'TESTING segmentweise - Achtung: Ergebnisse stimmen nicht mehr!'