# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:43:09 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""


import numpy as np
import math # für nan
import matplotlib.pyplot as plt
import pandas as pd
from flixStructure import *


def plotFlow(calc, aFlow_value, label, withPoints = True):    
  # Linie:
  plt.step(calc.timeSeries, aFlow_value, where = 'post', label = label)
  # Punkte dazu:
  if withPoints:
    # TODO: gleiche Farbe!
    # aStr = 'C' + str(i) + 'o'
    aStr = 'o'
    plt.plot(calc.timeSeries, aFlow_value, aStr)


# TODO: könnte man ggf. schöner mit dict-result machen bzw. sollte auch mit dict-result gehen!
def plotOn(mb, aVar_struct, var_label, y, plotSwitchOnOff = True):  
  try:
    plt.step(    mb.timeSeries, aVar_struct.on_        * y, ':',  where = 'post', label= var_label + '_On')
    if plotSwitchOnOff:
      try:
        plt.step(mb.timeSeries, aVar_struct.switchOn_  * y, '+',  where = 'post', label= var_label + '_SwitchOn')
        plt.step(mb.timeSeries, aVar_struct.switchOff_ * y, 'x',  where = 'post', label= var_label + '_SwitchOff')  
      except:
        pass
  except:
    pass

# # Input z.B. 'results_struct.KWK.Q_th.on' oder [KWK,'Q_th','on']
# def plotSegmentedValue(calc : cCalculation, results_struct_As_String_OR_keyList):    
#   if len(calc.segmentModBoxList) == 0 :
#     raise Exception 'Keine Segmente vorhanden!'
#   else :
#     for aModBox in calc.segmentModBoxList:
#       # aVal:
#       eval('aVal = aModBox.' + results_struct_As_String)
#       # aTimeSeries:
#       aTimeSeries = aModBox.timeSeriesWithEnd[:len(aVal)] # ggf. um 1 kürzen, wenn kein Speicherladezustand
#       plt.step(aTimeSeries, aVal, where = 'post')
  