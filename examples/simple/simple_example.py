##############################
## PreProcessing + Solving: ##
##############################

import simple_example_PreProcessing as example_pre

# Name der Rechnung:
nameOfCalc = example_pre.aCalc.nameOfCalc

if hasattr(example_pre, 'calcSegs'): 
  nameOfCalcSegs = example_pre.calcSegs.nameOfCalc
else:
  nameOfCalcSegs = None
print(nameOfCalc)

# ####################
# # PostProcessing: ##
# ####################

# ##### loading ######
import flixPostprocessing as flixPost
calc1_res = flixPost.flix_results(nameOfCalc)

if nameOfCalcSegs is not None:  
  calcSegs = flixPost.flix_results(nameOfCalcSegs)
else: 
  calcSegs = None

##### plotting ######

calc1_res.plotInAndOuts('Fernwaerme',stacked=True)
calc1_res.plotInAndOuts('Fernwaerme',stacked=True, outCompsAboveXAxis='Waermelast')
calc1_res.plotInAndOuts('Fernwaerme',stacked=False, outCompsAboveXAxis='Waermelast')
calc1_res.plotInAndOuts('Fernwaerme',stacked=True, plotAsPlotly = True, outCompsAboveXAxis = 'Waermelast')
calc1_res.plotInAndOuts('BHKW2',stacked=True)

calc1_res.plotShares(['Fernwaerme','Strom'], withoutStorage = True)
calc1_res.plotShares('Fernwaerme', withoutStorage = True, plotAsPlotly  = True, unit='kWh')

import matplotlib.pyplot as plt
from flixPlotHelperFcts import *

# Zeitreihe greifen:
print('Variante 1:')
print(calc1_res.results['Kessel']['Q_th']['val'])
print('Variante 2:')
print(calc1_res.results_struct.Kessel.Q_th.val)


# Beispiele von Zugriff:
sum(calc1_res.results_struct.globalComp.costs.operation.sum_TS)
calc1_res.results_struct.globalComp.costs.operation.sum

fuel = calc1_res.results_struct.Gastarif.Q_Gas.val * 0.04 - calc1_res.results_struct.Einspeisung.P_el.val * 0.7
print(fuel)
print(sum(fuel))


## sonstige Plots: ##

# Test abschnittsweise linear:
plt.plot(calc1_res.results_struct.BHKW2.P_el.val, calc1_res.results_struct.BHKW2.Q_th.val,'.',markersize=4,label='Q_th')
plt.plot(calc1_res.results_struct.BHKW2.P_el.val, calc1_res.results_struct.BHKW2.Q_fu.val,'.',markersize=4,label='Q_fu')
plt.legend()
plt.grid()
plt.show()


# Übersichtsplot:
import matplotlib.pyplot as plt


def uebersichtsPlot(aCalc):
  fig, ax = plt.subplots(figsize=(10, 5))
  plt.title(aCalc.label)
    
  plotFlow(aCalc, aCalc.results_struct.BHKW2.P_el.val,  'P_el')
  plotFlow(aCalc, aCalc.results_struct.BHKW2.Q_th.val,  'Q_th_BHKW')
  plotFlow(aCalc, aCalc.results_struct.Kessel.Q_th.val, 'Q_th_Kessel')
  
  plotOn(aCalc, aCalc.results_struct.Kessel.Q_th, 'Q_th_Kessel', -5)
  plotOn(aCalc, aCalc.results_struct.Kessel,      'Kessel'     , -10)
  plotOn(aCalc, aCalc.results_struct.BHKW2,       'BHKW '      , -15)
  
  plotFlow(aCalc, aCalc.results_struct.Waermelast.Q_th_Last.val, 'Q_th_Last')
  
  plt.plot(aCalc.timeSeries, aCalc.results_struct.globalComp.costs.operation.sum_TS, '--', label='costs (operating)') 
  
  if hasattr(aCalc.results_struct,'Speicher'):
    plt.step(aCalc.timeSeries, aCalc.results_struct.Speicher.Q_th_unload.val, where = 'post', label='Speicher_unload')
    plt.step(aCalc.timeSeries, aCalc.results_struct.Speicher.Q_th_load.val  , where = 'post', label='Speicher_load')
    plt.plot(aCalc.timeSeriesWithEnd, aCalc.results_struct.Speicher.charge_state   , label='charge_state')
  # plt.step(aCalc.timeSeries, aCalc.results_struct.Speicher., label='Speicher_load')
  plt.grid(axis='y')
  plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
  plt.show()

uebersichtsPlot(calc1_res)
if calcSegs is not None : uebersichtsPlot(calcSegs)

    
