##############################
## PreProcessing + Solving: ##
##############################

import example_complex_ModelAndSolve as example_pre

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
import flixOpt.flixPostprocessing as flixPost
# comp_colors = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
comp_colors = None
# https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express

calc1 = flixPost.flix_results(nameOfCalc, comp_colors = comp_colors)

#explizite Farbänderung
calc1.postObjOfStr('Waermelast').color = '#000000'

if nameOfCalcSegs is not None:  
  calcSegs = flixPost.flix_results(nameOfCalcSegs)
else: 
  calcSegs = None

##### plotting ######

fig1 = calc1.plotInAndOuts('Fernwaerme',stacked=True)
fig1.savefig('results/test1')
fig2 = calc1.plotInAndOuts('Fernwaerme',plotAsPlotly = True)
fig2.write_html('results/test2.html')

fig = calc1.plotInAndOuts('Fernwaerme',stacked=True, flowsAboveXAxis='Waermelast', sortBy='Waermelast')
fig.show()
fig = calc1.plotInAndOuts('Fernwaerme',stacked=True, flowsAboveXAxis='Waermelast')
fig.show()
fig = calc1.plotInAndOuts('Fernwaerme',stacked=False, flowsAboveXAxis='Waermelast')
fig.show()
fig = calc1.plotInAndOuts('Fernwaerme',stacked=True, plotAsPlotly = True, flowsAboveXAxis = 'Waermelast', sortBy='Waermelast')
fig.show()
fig = calc1.plotInAndOuts('Fernwaerme',stacked=True, plotAsPlotly = True, flowsAboveXAxis = 'Waermelast')
fig.show()
fig = calc1.plotInAndOuts('BHKW2',stacked=True, inFlowsPositive = False)
fig.show()

fig = calc1.plotShares(['Fernwaerme','Strom'], withoutStorage = True)
fig.show()
fig = calc1.plotShares('Fernwaerme', withoutStorage = True, plotAsPlotly  = True, unit='kWh')
fig.show()

import matplotlib.pyplot as plt
from flixOpt.flixPlotHelperFcts import *

# Zeitreihe greifen:
print('Variante 1:')
print(calc1.results['Kessel']['Q_th']['val'])
print('Variante 2:')
print(calc1.results_struct.Kessel.Q_th.val)


# Beispiele von Zugriff:
sum(calc1.results_struct.globalComp.costs.operation.sum_TS)
calc1.results_struct.globalComp.costs.operation.sum

fuel = calc1.results_struct.Gastarif.Q_Gas.val * 0.04 - calc1.results_struct.Einspeisung.P_el.val * 0.7
print(fuel)
print(sum(fuel))


## sonstige Plots: ##

# Test abschnittsweise linear:
plt.plot(calc1.results_struct.BHKW2.P_el.val, calc1.results_struct.BHKW2.Q_th.val,'.',markersize=4,label='Q_th')
plt.plot(calc1.results_struct.BHKW2.P_el.val, calc1.results_struct.BHKW2.Q_fu.val,'.',markersize=4,label='Q_fu')
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

uebersichtsPlot(calc1)
if calcSegs is not None : uebersichtsPlot(calcSegs)

calc1.to_csv('Fernwaerme', 'results/FW.csv')
    
# loading yaml-datei:
with open(calc1.filename_infos,'rb') as f:
    infos = yaml.safe_load(f)
