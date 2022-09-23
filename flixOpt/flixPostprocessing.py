# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:05:50 2022

@author: Panitz
"""

import pickle
import yaml
import flixOptHelperFcts as helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # für Plots im Postprocessing
import matplotlib.dates as mdates

class cFlow_post():
  def __init__(self,aDescr,flixResults):
    self.label = aDescr['label']
    self.bus   = aDescr['bus']
    self.comp  = aDescr['comp']
    self.descr = aDescr
    self.flixResults = flixResults
    # Richtung:    
    self.isInputInComp = aDescr['isInputInComp']    
    if self.isInputInComp:
      self.from_node = self.bus
      self.to_node = self.comp
    else:
      self.from_node = self.comp
      self.to_node = self.bus
    
  def extractResults(self, allResults):
    self.results = allResults[self.comp][self.label]
    self.results_struct = helpers.createStructFromDictInDict(self.results)
  def getFlowHours(self):
    flowHours = sum(self.results['val']* self.flixResults.dtInHours)
    return flowHours
  
  def getLoadFactor(self, small=1e-2):
   loadFactor = None
   if ('invest' in self.results.keys()) and ('nominal_val' in self.results['invest'].keys()):
     flowHours = self.getFlowHours()
     #  loadFactor = Arbeit / Nennleistung / Zeitbereich = kWh / kW_N / h 
     nominal_val = self.results['invest']['nominal_val']
     if nominal_val < small:
       loadFactor = None 
     else:
       loadFactor = flowHours / self.results['invest']['nominal_val'] / self.flixResults.dtInHours_tot
   return loadFactor
 
  def belongToStorage(self):
    if 'isStorage' in self.flixResults.infos_system['components'][self.comp].keys():
      return self.flixResults.infos_system['components'][self.comp]['isStorage']
    else:
      return False
    
class flix_results():
  def __init__(self, nameOfCalc, results_folder = None, ): #,timestamp = None):

    self.label = nameOfCalc
      # 'z.B.' 2022-06-14_Sim1_gurobi_SolvingInfos

    filename_infos = 'results/' + nameOfCalc + '_solvingInfos.yaml'
    filename_data = 'results/' + nameOfCalc + '_data.pickle'


    with open(filename_infos,'rb') as f:
      self.infos = yaml.safe_load(f)
      self.infos_system = self.infos['system_description']

    with open(filename_data,'rb') as f:
      self.results = pickle.load(f)          
    self.results_struct = helpers.createStructFromDictInDict(self.results)
    self.buses = self.__getBuses()
    self.comps = self.__getComponents()
    
    # Zeiten:
    self.timeSeries = self.results['time']['timeSeries']
    self.timeSeriesWithEnd = self.results['time']['timeSeriesWithEnd']
    self.dtInHours = self.results['time']['dtInHours']
    self.dtInHours_tot = self.results['time']['dtInHours_tot']
    
    self.flows = self.__getAllFlows()
    
  def __getBuses(self):
    try :
      return list(self.infos_system['buses'].keys())
    except:
      raise Exception('buses nicht gefunden')
    
  
  def __getComponents(self):
    try :
      return list(self.infos_system['components'].keys())
    except:
      raise Exception('components nicht gefunden')
    
  def __getAllFlows(self):
    flows = []
    flow_list = self.infos_system['flows']
    # todo: Auslesen der Flows könnte man sich auch einfacher machen über Flow-Liste in system_description
    for flow_Descr in flow_list:      
      aFlow = cFlow_post(flow_Descr,self)
      aFlow.extractResults(self.results)
      flows.append(aFlow)    
    return flows
  
  def getFlowsOf(self, node, node2=None):
    inputs_node = []
    outputs_node = []
    
    if node not in (self.buses + self.comps):
      raise Exception('node \'' + str(node) + '\' not in buses or comps')
    if node2 not in (self.buses + self.comps) and (node2 is not None):
      raise Exception('node2 \'' + str(node2) + '\' not in buses or comps')
    
    for flow in self.flows:
      if node in [flow.bus, flow.comp] \
        and ((node2 is None) or (node2 in [flow.bus, flow.comp])):
        if node == flow.to_node:
          inputs_node.append(flow)
        elif node == flow.from_node:
          outputs_node.append(flow)
        else:
          raise Exception('node ' + node + ' not in flow.from_node or flow.to_node' )
    
    return (inputs_node, outputs_node)

  @staticmethod
  # check if SeriesValues should be shown
  def isGreaterMinFlowHours(aFlowValues,dtInHours,minFlowHours):
    # absolute Summe, damit auch negative Werte gezählt werden:
    absFlowHours = sum(abs(aFlowValues * dtInHours))
    # print(absFlowHours)
    return absFlowHours > minFlowHours
    

  @staticmethod
  # für plot:    
  def __get_Values_As_DataFrame(flows,timeSeries,dtInHours, minFlowHours):
    y = pd.DataFrame(index = timeSeries) # letzten Zeitschritt vorerst 


    

    for aFlow in flows:        
      # assert aFlow.comp.label in results.keys(), '!Keine Werte für Flow "' + aFlow.label_full + '" vorhanden!'
      # y[aFlow.comp.label + '.' + aFlow.label] =   aFlow.mod.var_val.getResult() # ! positiv!                    
      values = aFlow.results['val'] # 
      # print(aFlow.label_full)
      values[np.logical_and(values<0, values>-1e-5)] = 0 # negative Werte durch numerische Auflösung löschen 
      assert (values>=0).all(), 'Warning, Zeitreihen '+ aFlow.label_full +' in inputs enthalten neg. Werte -> Darstellung Graph nicht korrekt'
      
      # if inOrOut = 'in':
      #   sign = +1
      # elif inOrOut = 'out':
      #   sign = -1
      # else:
      #   raise Exception('not defined')                      
      if flix_results.isGreaterMinFlowHours(values, dtInHours, minFlowHours): # nur wenn gewisse FlowHours-Sum überschritten
          y[aFlow.comp + '.' + aFlow.label] = + values # ! positiv!
    return y
  
  def getLoadFactorOfComp(self,aComp):
    (in_flows, out_flows) = self.getFlowsOf(aComp)
    for aFlow in (in_flows+out_flows):
      loadFactor = aFlow.getLoadFactor()
      if loadFactor is not None:
        # Abbruch Schleife und return:
        return loadFactor 
    return None
         
  def getLoadFactorsOfComps(self,withoutNone = True):
    loadFactors = {}
    comps = self.__getComponents()
    for comp in comps:
      loadFactor = self.getLoadFactorOfComp(comp)
      if loadFactor is not None:
        loadFactors[comp] = loadFactor
    return loadFactors
  
  def getFlowHours(self,busOrComp,useInputs = True,skipZeros=True):
    small = 1e5
    FH = {}
    (in_flows, out_flows) = self.getFlowsOf(busOrComp)
    if useInputs:
      flows =in_flows 
    else:
      flows = out_flows
    for aFlow in flows:
      flowHours = aFlow.getFlowHours() 
      if flowHours > small:
        FH [aFlow.comp + '.' + aFlow.label] = aFlow.getFlowHours() 
    return FH
  
  
  # def plotFullLoadHours(self):
  #   FLH = self.getLoadFactor
    
    
      
  # def plotFullLoadHours(self):
  #   comps = self.__getComponents()
  #   for comp in comps:
  #     getFlowHours
  #     self.getFullLoadHoursOfComp(comp)
      
  
  def plotShares(self,busOrComponent, useInputs=True, withoutStorage = True,  minSum=.1, othersMax_rel=0.05, plotAsPlotly = False, title = None, unit = 'FlowHours'):      
    (in_flows, out_flows) = self.getFlowsOf(busOrComponent)
    if useInputs:
      flows =in_flows 
    else:
      flows = out_flows
    
    
    # delete flows which belong to storage (cause they disturbing the plot):
    if withoutStorage:
      # Umsetzung not nice, aber geht!
      allowed_i = [] 
      for i in range(len(flows)):        
        aFlow = flows[i]        
        if not aFlow.belongToStorage():
          allowed_i.append(i)
      flows = [flows[i] for i in allowed_i] # Gekürzte liste
      

    sums = np.array([])
    labels = []
    totalSum = 0
    for aFlow in flows:
      totalSum +=sum(aFlow.results['val'])
      
    others_Sum = 0
    for aFlow in flows:
      aSum = sum(aFlow.results['val'])
      if aSum >minSum:
        if aSum/totalSum < othersMax_rel:
          others_Sum += aSum
        else:
          sums=np.append(sums,aSum)
          labels.append(aFlow.comp + '.' + aFlow.label)
    
    if others_Sum >0:
      sums = np.append(sums,others_Sum)
      labels.append('others')

    aText = "total: {:.0f}".format(sum(sums)) + ' ' + unit 

    if title is None:
        title=busOrComponent
        if useInputs:
          title+= ' (supply)'
        else:
          title+= ' (usage)'
        

    

    def plot_matplotlib(sums, labels, title, aText):
        fig = plt.figure()
        ax = fig.add_subplot()
        # ax.title(busOrComponent)
        plt.title(title)
        plt.pie(sums/sum(sums), labels = labels)            
        fig.text(0.95, 0.05, aText,
              verticalalignment='top', horizontalalignment='center',
              transform=ax.transAxes,
              color='black', fontsize=10)
        # ax.text(0.95, 0.98, aText,
        #       verticalalignment='top', horizontalalignment='right',
        #       transform=ax.transAxes,
        #       color='black', fontsize=10)
        plt.show()
    
    def plot_plotly(sums, labels,title, aText):            
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Pie(labels=labels, values=sums)])
        fig.update_layout(title_text = title,
                          annotations = [dict(text=aText, x=0.95, y=0.05, font_size=20, align = 'right', showarrow=False)],
                          )
        fig.show()
        
    if plotAsPlotly:
      plot_plotly    (sums, labels, title, aText)
    else:
      plot_matplotlib(sums, labels, title, aText)
                       
      
  def plotInAndOuts(self, busOrComponent, stacked = False, renderer='browser', minFlowHours=0.1, plotAsPlotly = False, title = None):      
    '''      
    Parameters
    ----------
    busOrComponent : TYPE
        DESCRIPTION.
    stacked : TYPE, optional
        DESCRIPTION. The default is False.
    renderer : TYPE, optional
        DESCRIPTION. The default is 'browser'.
    minFlowHours : TYPE, optional
        DESCRIPTION. The default is 0.1.
    plotAsPlotly : boolean, optional
    
    title : str, optional
        if None, then automatical title is used

    '''
    if not (busOrComponent in self.results.keys()):
      raise Exception(str(busOrComp) + 'is no valid bus or component name')

    # minFlowHours -> min absolute sum of Flows for Showing curve
    # renderer     -> 'browser', 'svg',...
    
    import plotly.io as pio            
    pio.renderers.default = renderer # 'browser', 'svg',...

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
       
    
    # Dataframe mit Inputs (+) und Outputs (-) erstellen:
    timeSeries = self.timeSeriesWithEnd[0:-1] # letzten Zeitschritt vorerst weglassen
    (in_flows, out_flows) = self.getFlowsOf(busOrComponent)
    # Inputs:
    y_in = self.__get_Values_As_DataFrame(in_flows, timeSeries, self.dtInHours, minFlowHours)
    # Outputs; als negative Werte interpretiert:
    y_out = -1 * self.__get_Values_As_DataFrame(out_flows,timeSeries, self.dtInHours, minFlowHours)

    
    # if hasattr(self, 'excessIn')  and (self.excessIn is not None):
    if 'excessIn' in self.results[busOrComponent].keys():
      # in and out zusammenfassen:
      
      excessIn = self.results[busOrComponent]['excessIn'] 
      excessOut = - self.results[busOrComponent]['excessOut']
      
      if flix_results.isGreaterMinFlowHours(excessIn, self.dtInHours, minFlowHours):        
        y_in['excess_in']   = excessIn
      if flix_results.isGreaterMinFlowHours(excessOut, self.dtInHours, minFlowHours):        
        y_out['excess_out'] = excessOut
            
      
    def appendEndTimeStep(y):   
      # hänge noch einen Zeitschrtt mit gleichen Werten an (withEnd!) damit vollständige Darstellung
      lastRow = y.iloc[-1] # kopiere aktuell letzte
      lastTimeStep = self.timeSeriesWithEnd[-1] # withEnd
      lastRow = lastRow.rename(lastTimeStep) # Index ersetzen -> letzter Zeitschritt als index        
      y=y.append(lastRow) # anhängen
      return y


    y_in = appendEndTimeStep(y_in)
    y_out = appendEndTimeStep(y_out)

    # wenn title nicht gegeben
    if title is None:
        title = busOrComponent + ': '+ ' in (+) and outs (-)' + ' [' + self.label + ']'
    yaxes_title = 'Flow'
    yaxes2_title = 'charge state'


    def plotY_plotly(y_pos, y_neg, title, yaxes_title, yaxes2_title):

      ## Flows:
      # fig = go.Figure()
      # Create figure with secondary y-axis
      fig = make_subplots(specs=[[{"secondary_y": True}]])
      
      # def plotlyPlot(y_in, y_out, stacked, )
      # input:
      for column in y_in.columns:
          # if isGreaterMinAbsSum(y_in[column]):
          if stacked :
            fig.add_trace(go.Scatter(x=y_in.index, y=y_in[column],stackgroup='one',line_shape ='hv',name=column))
          else:
            fig.add_trace(go.Scatter(x=y_in.index, y=y_in[column],line_shape ='hv',name=column))
      # output:
      for column in y_out.columns:
          # if isGreaterMinAbsSum(y_out[column]):
          if stacked :
            fig.add_trace(go.Scatter(x=y_out.index, y=y_out[column],stackgroup='two',line_shape ='hv',name=column))
          else:
            fig.add_trace(go.Scatter(x=y_out.index, y=y_out[column],line_shape ='hv',name=column))
       
      # ## Speicherverlauf auf Sekundärachse:
      # # Speicher finden:
      # setOfStorages = set()
      # # for aFlow in self.inputs + self.outputs:
      # for acomp in self.modBox.es.allMEsOfFirstLayerWithoutFlows:
      #   if acomp.__class__.__name__ == 'cStorage': # nicht schön, da cStorage hier noch nicht bekannt
      #     setOfStorages.add(acomp)      
      # for aStorage in setOfStorages:
      #   chargeState = aStorage.mod.var_charge_state.getResult()
      #   fig.add_trace(go.Scatter(x=timeSeriesWithEnd, y=chargeState, name=aStorage.label+'.chargeState',line_shape='linear',line={'dash' : 'dash'} ),secondary_y = True)
        
  
      # fig.update_layout(title_text = title,
      #                   xaxis_title = 'Zeit',
      #                   yaxis_title = 'Leistung [kW]')
      #                   # yaxis2_title = 'charge state')      
      fig.update_xaxes(title_text="Zeit")
      fig.update_yaxes(title_text=yaxes_title, secondary_y=False)
      fig.update_yaxes(title_text=yaxes2_title, secondary_y=True)      
      fig.update_layout(title=title)
      fig.show()
    
    def plotY_matplotlib(y_pos, y_neg, title, yaxes_title, yaxes2_title):
      # Verschmelzen:
      y = pd.concat([y_pos,y_neg],axis=1)
            
      fig, ax = plt.subplots(figsize = (18,10)) 
    
      # gestapelt:
      if stacked :               
        helpers.plotStackedSteps(ax, y) # legende here automatically
        plt.legend(fontsize=22, loc="upper center",  bbox_to_anchor=(0.5, -0.2), markerscale=2,  ncol=3, frameon=True, fancybox= True, shadow=True)
      
      # normal:
      else:
        y.plot( drawstyle='steps-post', figsize=(18,10))                                            ### größerer Plot, 16:9 style                            
        plt.legend( loc="upper center",  bbox_to_anchor=(0.5, -0.2), fontsize=22, ncol=3, frameon=True, shadow= True, fancybox=True) #####David Test  Legende unter Plot
                        
      fig.autofmt_xdate()
     
      xfmt = mdates.DateFormatter('%d-%m')
      ax.xaxis.set_major_formatter(xfmt)
      
      plt.title(title, fontsize= 30)
      plt.xlabel('Zeit - Woche [h]', fontsize = 'xx-large')                                                 ### x-Achsen-Titel                     
      plt.ylabel(yaxes_title ,fontsize = 'xx-large')                                            ### y-Achsen-Titel  = Leistung immer
      plt.grid()
      plt.show()      
      
      
    
    if plotAsPlotly:
      plotY_plotly(y_in, y_out, title, yaxes_title, yaxes2_title)
    else:
      plotY_matplotlib(y_in, y_out, title, yaxes_title, yaxes2_title)
                       
              

# self.results[]
## ideen: kosten-übersicht, JDL, Torten-Diag für Busse