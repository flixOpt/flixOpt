# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:05:50 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import pickle
import yaml
from . import flixOptHelperFcts as helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # für Plots im Postprocessing
import matplotlib.dates as mdates

class cFlow_post():


    @property
    def color(self):
        # interaktiv, falls component explizit andere Farbe zugeordnet
        return self._getDefaultColor()        

    @property
    def label_full(self):
        return self.comp + '__' + self.label

    def __init__(self,aDescr,flixResults):
        self.label = aDescr['label']
        self.bus   = aDescr['bus']
        self.comp  = aDescr['comp']
        if "group" in aDescr:
            self.group = aDescr['group']
        self.descr = aDescr
        self.flixResults = flixResults
        self.comp_post = flixResults.postObjOfStr(self.comp)
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
    def _getDefaultColor(self):
        return self.comp_post.color
        
    
class cCompOrBus_post():
    def __init__(self,label, aDescr, flixResults, color = None):
        self.label = label
        self.type  = aDescr['class']
        if "group" in aDescr:
            self.group = aDescr['group']
        self.descr = aDescr
        self.flixResults = flixResults
        self.color = color
    
class flix_results():
    def __init__(self, nameOfCalc, results_folder = 'results', comp_colors = None):
        '''
        postprocessing: Create this object to load a solved optimization-calculation into workspace

        Parameters
        ----------
        nameOfCalc : str
            name of calculation which should be loaded
        results_folder : str
            location of the result-files
        comp_colors : list of colors, optional
            List of colors, which will be used for plotting. If None, than default color list is used.            
            
        '''
        
        
        self.label = nameOfCalc
        self.comp_colors = comp_colors
        # default value:
        if self.comp_colors == None:
            import plotly.express as px
            self.comp_colors = (px.colors.qualitative.Light24 + px.colors.qualitative.Bold +
                                px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 +
                                px.colors.qualitative.Light24 * 10)
            # see: https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
            
        # 'z.B.' 2022-06-14_Sim1_gurobi_SolvingInfos
        self.filename_infos = results_folder + '/' + nameOfCalc + '_solvingInfos.yaml'
        self.filename_data = results_folder + '/' + nameOfCalc + '_data.pickle'
        
    
        with open(self.filename_infos,'rb') as f:
            self.infos = yaml.safe_load(f)
            self.infos_system = self.infos['system_description']
    
        with open(self.filename_data,'rb') as f:
            self.results = pickle.load(f)          
        self.results_struct = helpers.createStructFromDictInDict(self.results)
        
        # list of str:
        self.buses = self.__getBuses()
        self.comps = self.__getComponents()
        # list of post_obj:
        self.bus_posts = self.__getAllBuses()   
        self.comp_posts = self.__getAllComps()
        self.flows = self.__getAllFlows()
        
        # Zeiten:
        self.timeSeries = self.results['time']['timeSeries']
        self.timeSeriesWithEnd = self.results['time']['timeSeriesWithEnd']
        self.dtInHours = self.results['time']['dtInHours']
        self.dtInHours_tot = self.results['time']['dtInHours_tot']
      
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
        for flow_Descr in flow_list:      
            aFlow = cFlow_post(flow_Descr, self)
            aFlow.extractResults(self.results)
            flows.append(aFlow)    
        return flows

    
    def __getAllComps(self):
        comps = []
        comp_dict = self.infos_system['components']
        
        myColorIter = iter(self.comp_colors)
        for label,descr in comp_dict.items():      
            try: 
                aComp = cCompOrBus_post(label, descr, self, color = next(myColorIter))
            except StopIteration:
                raise Exception('too less colors defined for printing!')                
            # TODO: extract Results similar to cFlow_post
            comps.append(aComp)
        return comps
    
    def __getAllBuses(self):
        buses = []
        bus_dict = self.infos_system['buses']
        for label,descr in bus_dict.items():      
            aBus = cCompOrBus_post(label, descr, self)
            buses.append(aBus)
        return buses
        
    
    def getFlowsOf(self, node, node2=None):
        '''
        

        Parameters
        ----------
        node : str
            component or bus where flow is linked to
        node2 : str, optional
            component or bus where flow is linked to

        Returns
        -------
        inputs_node : array
            flows, which are input to node (and output to node2)
        outputs_node : array
            flows, which are output to node (and input to node2)

        '''      
        
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
  
    def postObjOfStr(self, aStr):
        thePostObj = None
        for aPostObj in self.comp_posts + self.bus_posts:
            if aPostObj.label == aStr:
                thePostObj = aPostObj        
        return thePostObj
    
    
    @staticmethod
    # check if SeriesValues should be shown
    def isGreaterMinFlowHours(aFlowValues,dtInHours,minFlowHours):
        # absolute Summe, damit auch negative Werte gezählt werden:
        absFlowHours = sum(abs(aFlowValues * dtInHours))
        # print(absFlowHours)
        return absFlowHours > minFlowHours
      
  
    
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
        
    
    def plotShares(self, busesOrComponents, useInputs=True, withoutStorage = True, minSum=.1, othersMax_rel=0.05, plotAsPlotly = False, title = None, unit = 'FlowHours'):      
        '''     
        plots FlowHours in pie-plot
        
        Parameters
        ----------
        busesOrComponents : str or list of str
            DESCRIPTION.
        useInputs : TYPE, optional
            DESCRIPTION. The default is True.
        withoutStorage : TYPE, optional
            DESCRIPTION. The default is True.
        minSum : TYPE, optional
            DESCRIPTION. The default is .1.
        othersMax_rel : TYPE, optional
            DESCRIPTION. The default is 0.05.
        plotAsPlotly : TYPE, optional
            DESCRIPTION. The default is False.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        unit : TYPE, optional
            DESCRIPTION. The default is 'FlowHours'.
        Returns
        -------
        figure
        '''

        # if not a list of str yet, transform to list:
        if isinstance(busesOrComponents, str):
            busesOrComponents =  [busesOrComponents]

        in_flows_all = []            
        out_flows_all = []
        for busOrComponent in busesOrComponents:          
            (in_flows, out_flows) = self.getFlowsOf(busOrComponent)
            in_flows_all += in_flows
            out_flows_all += out_flows
            
        if useInputs:
            flows = in_flows_all
        else:
            flows = out_flows_all
        
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
        colors = []
        
        # total flowhours of chosen balance:
        totalSum = 0
        for aFlow in flows:
          totalSum +=sum(aFlow.results['val'] * self.dtInHours) # Flow-Hours
        
        # single shares of total flowhours:
        others_Sum = 0
        for aFlow in flows:
          aSum = sum(aFlow.results['val']* self.dtInHours) # flow-hours
          if aSum >minSum:
            if aSum/totalSum < othersMax_rel:
              others_Sum += aSum
            else:
              sums=np.append(sums,aSum)
              labels.append(aFlow.comp + '.' + aFlow.label)
              colors.append(aFlow.color)
              
        
        if others_Sum >0:
          sums = np.append(sums,others_Sum)
          labels.append('others')
          colors.append('#AAAAAA')# just a grey
        
        aText = "total: {:.0f}".format(sum(sums)) + ' ' + unit 
        
        if title is None:
            title=",".join(busesOrComponents)
            if useInputs:
              title+= ' (supply)'
            else:
              title+= ' (usage)'
            
        
        
        
        def plot_matplotlib(sums, labels, title, aText):
            fig = plt.figure()
            ax = fig.add_subplot()
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
            # plt.show()            
            return fig
        
        def plot_plotly(sums, labels,title, aText, colors):            
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(labels=labels, values=sums, marker_colors = colors)])
            fig.update_layout(title_text = title,
                              annotations = [dict(text=aText, x=0.95, y=0.05, font_size=20, align = 'right', showarrow=False)],
                              )
            
            # fig.show()
            return fig
            
        if plotAsPlotly:
            fig = plot_plotly    (sums, labels, title, aText, colors)
        else:
            fig = plot_matplotlib(sums, labels, title, aText)
        return fig
                           
    def to_csv(self, busOrComponent, filename, sep ='\t'):
        '''
        saves flow-values to csv # TODO: TEMPORARY function only!

        Parameters
        ----------
        busOrComponent : str
            flows linked to this bus or component are chosen
        filename : str
            DESCRIPTION.
        sep : str
            separator, i.g. '/t', ';', ...
        '''
        
        
        (in_flows, out_flows) = self.getFlowsOf(busOrComponent)  

        df = pd.DataFrame(index=self.timeSeries)
        for flow in in_flows + out_flows:
            flow:cFlow_post 
            flow.label_full
            df[flow.label_full] = flow.results['val']
        df.to_csv(filename, sep = sep)
        
    def plotInAndOuts(self, 
                      busOrComponent, 
                      stacked = False, 
                      renderer='browser', 
                      minFlowHours=0.1,
                      plotAsPlotly = False, 
                      title = None, 
                      flowsAboveXAxis=None,
                      sortBy = None,
                      inFlowsPositive = True):
        '''      
        Parameters
        ----------
        busOrComponent : TYPE
            DESCRIPTION.
        stacked : TYPE, optional
            DESCRIPTION. The default is False.
        renderer : 'browser', 'svg',...
        
        minFlowHours : TYPE, optional
            min absolute sum of Flows for Showing curve. The default is 0.1.
        plotAsPlotly : boolean, optional
        
        title : str, optional
            if None, then automatical title is used    
            
        flowsAboveXAxis : list of str (components, buses)
            End-Nodes (i.e. components, flows) of flows, 
            which should be shown separately above x-Axis.
            
            i.g. flowsAboveXAxis = ["heat_load", "absorptionChiller"]
            
        sortBy : component or None, optional    
            Component-Flow which should be used for sorting the timeseries ("Jahresdauerlinie")
            
        inFlowsPositive: boolean, default: TRUE
            wether inFlows or outFlows should be positive (above x-Axis)
            
            
        Return
        ------
        
        figure-object
        '''
        
        if not (busOrComponent in self.results.keys()):
            raise Exception(str(busOrComponent) + 'is no valid bus or component name')
        
        import plotly.io as pio            
        pio.renderers.default = renderer # 'browser', 'svg',...
    
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        (in_flows, out_flows) = self.getFlowsOf(busOrComponent)           
               
        # sorting:
        if sortBy is not None:
            # find right flow:
            (ins, outs) = self.getFlowsOf(busOrComponent,sortBy)
            flowForSort = (ins+outs)[0] # dirty!
            # find index sequence
            indexSeq = np.argsort(flowForSort.results['val']) # ascending
            indexSeq = indexSeq[::-1] # descending
        else:
            indexSeq = None
            
        # extract flows eplicitly above x-Axis:
        flowObjects_above_x_axis = []
        if flowsAboveXAxis is not None:
            for flow in out_flows:
                if flow.to_node in flowsAboveXAxis:
                    out_flows.remove(flow)
                    flowObjects_above_x_axis.append(flow)
            for flow in in_flows:
                if flow.from_node in flowsAboveXAxis:
                    in_flows.remove(flow)
                    flowObjects_above_x_axis.append(flow)

        
        # Inputs:
        if inFlowsPositive:
            pos_flows = in_flows
            neg_flows = out_flows
        else:
            pos_flows = out_flows
            neg_flows = in_flows        
            
        y_pos, y_pos_colors,  = self._get_FlowValues_As_DataFrame(pos_flows, self.timeSeriesWithEnd, self.dtInHours, minFlowHours)
        # Outputs; als negative Werte interpretiert:
        y_neg, y_neg_colors = self._get_FlowValues_As_DataFrame(neg_flows,self.timeSeriesWithEnd, self.dtInHours, minFlowHours)
        y_neg = -1 * y_neg 
        # Outputs above X-Axis:
        y_neg_aboveX, y_above_colors = self._get_FlowValues_As_DataFrame(flowObjects_above_x_axis,self.timeSeriesWithEnd, self.dtInHours, minFlowHours)

        # append excessValues:
        if 'excessIn' in self.results[busOrComponent].keys():
            # in and out zusammenfassen:
            
            excessIn = self.results[busOrComponent]['excessIn'] 
            excessOut = - self.results[busOrComponent]['excessOut']
            
            if flix_results.isGreaterMinFlowHours(excessIn, self.dtInHours, minFlowHours):        
                y_pos['excess_in'] = excessIn
                y_pos_colors.append('#FF0000')
            if flix_results.isGreaterMinFlowHours(excessOut, self.dtInHours, minFlowHours):        
                y_neg['excess_out'] = excessOut
                y_neg_colors.append('#FF0000')
            
        # 1. sorting(if "sortBy") AND
        # 2. appending last index (for visualizing last bar (last timestep-width) in plot)
        listOfDataFrames = (y_pos, y_neg, y_neg_aboveX)
        y_pos = self._sortDataFramesAndAppendLastStep(y_pos, self.timeSeriesWithEnd, self.dtInHours, indexSeq=indexSeq)        
        y_neg = self._sortDataFramesAndAppendLastStep(y_neg, self.timeSeriesWithEnd, self.dtInHours, indexSeq=indexSeq)        
        y_neg_aboveX = self._sortDataFramesAndAppendLastStep(y_neg_aboveX, self.timeSeriesWithEnd, self.dtInHours, indexSeq=indexSeq)        
    
        # wenn title nicht gegeben
        if title is None:
            title = busOrComponent + ': '+ ' in (+) and outs (-)' + ' [' + self.label + ']'
        yaxes_title = 'Flow'
        yaxes2_title = 'charge state'
    
    
        def plotY_plotly(y_pos, y_neg, y_pos_separat, title, yaxes_title, yaxes2_title, y_pos_colors, y_neg_colors, y_above_colors):
    
            ## Flows:
            # fig = go.Figure()
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # def plotlyPlot(y_pos, y_neg, stacked, )
            # input:
            y_pos_colors = iter(y_pos_colors)
            y_neg_colors = iter(y_neg_colors)
            y_above_colors = iter(y_above_colors)
            for column in y_pos.columns:
                aColor = next(y_pos_colors)
                # if isGreaterMinAbsSum(y_pos[column]):
                if stacked :
                    fig.add_trace(go.Scatter(x=y_pos.index, y=y_pos[column], stackgroup='one', line_shape ='hv', name=column, line_color=aColor))
                else:
                    fig.add_trace(go.Scatter(x=y_pos.index, y=y_pos[column], line_shape='hv', name=column, line_color=aColor))
            # output:
            for column in y_neg.columns:
                aColor = next(y_neg_colors)
                # if isGreaterMinAbsSum(y_neg[column]):
                if stacked :
                    fig.add_trace(go.Scatter(x=y_neg.index, y=y_neg[column],stackgroup='two',line_shape ='hv',name=column, line_color = aColor))
                else:
                    fig.add_trace(go.Scatter(x=y_neg.index, y=y_neg[column],line_shape ='hv',name=column, line_color = aColor))
            
            # output above x-axis:
            for column in y_pos_separat:
                aColor = next(y_above_colors)
                fig.add_trace(go.Scatter(x=y_pos_separat.index, y=y_pos_separat[column],line_shape ='hv',line=dict(dash='dash', width = 4) , name=column, line_color = aColor))
            
            
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
            
            # fig.show()
            
            return fig

        def plotY_matplotlib(y_pos, y_neg, y_pos_separat, title, yaxes_title, yaxes2_title):
            # Verschmelzen:
            y = pd.concat([y_pos,y_neg],axis=1)
                  
            fig, ax = plt.subplots(figsize = (18,10)) 
            
            # separate above x_axis
            if len(y_pos_separat.columns) > 0 :                    
                for column in y_pos_separat.columns:
                    ax.plot(y_pos_separat.index, y_pos_separat[column], '--', drawstyle='steps-post', linewidth=3, label = column)

            # gestapelt:
            if stacked :               
                helpers.plotStackedSteps(ax, y) # legende here automatically                
            # normal:
            else:
                y.plot(drawstyle='steps-post', ax=ax)
                

            plt.legend(fontsize=22, loc="upper center",  bbox_to_anchor=(0.5, -0.2), markerscale=2,  ncol=3, frameon=True, fancybox= True, shadow=True)
            # plt.legend(loc="upper center",  bbox_to_anchor=(0.5, -0.2), fontsize=22, ncol=3, frameon=True, shadow= True, fancybox=True) 

                              
            fig.autofmt_xdate()
           
            xfmt = mdates.DateFormatter('%d-%m')
            ax.xaxis.set_major_formatter(xfmt)
            
            plt.title(title, fontsize= 30)
            plt.xlabel('Zeit - Woche [h]', fontsize = 'xx-large')                                                 ### x-Achsen-Titel                     
            plt.ylabel(yaxes_title ,fontsize = 'xx-large')                                            ### y-Achsen-Titel  = Leistung immer
            plt.grid()
            # plt.show()

            return fig
            
          
        if plotAsPlotly:
            fig = plotY_plotly(y_pos, y_neg, y_neg_aboveX, title, yaxes_title, yaxes2_title,
                         y_pos_colors, y_neg_colors, y_above_colors)
        else:
            fig = plotY_matplotlib(y_pos, y_neg, y_neg_aboveX, title, yaxes_title, yaxes2_title)
        
        return fig
            

    @staticmethod
    # get values of flow as dataframe and belonging colors:    
    def _get_FlowValues_As_DataFrame(flows, timeSeriesWithEnd ,dtInHours, minFlowHours):
        numericalZero = -1e-4
        # Dataframe mit Inputs (+) und Outputs (-) erstellen:
        y = pd.DataFrame() # letzten Zeitschritt vorerst weglassen
        y.index = timeSeriesWithEnd[0:-1] # letzten Zeitschritt vorerst weglassen       
        y_color = []
        # Beachte: hier noch nicht als df-Index, damit sortierbar
        for aFlow in flows:        
            values = aFlow.results['val'] # 
            values[np.logical_and(values<0, values>numericalZero)] = 0 # negative Werte durch numerische Auflösung löschen 
            assert (values>=numericalZero).all(), 'Warning, Zeitreihen '+ aFlow.label_full +' in inputs enthalten neg. Werte -> Darstellung Graph nicht korrekt'
                                
            if flix_results.isGreaterMinFlowHours(values, dtInHours, minFlowHours): # nur wenn gewisse FlowHours-Sum überschritten
                y[aFlow.label_full] = + values # ! positiv!
                y_color.append(aFlow.color)
        return y, y_color

    @staticmethod 
    def _sortDataFramesAndAppendLastStep(y, timeSeriesWithEnd, dtInHours, indexSeq = None):       
        # add index (for steps-graph)
        if indexSeq is not None: 
            y['dtInHours'] = dtInHours
            # sorting:
            y = y.iloc[indexSeq]
            # index:
            indexWithEnd = np.append([0], np.cumsum(y['dtInHours'].values))
            del y['dtInHours']            
            # Index überschreiben:
            y.index = indexWithEnd[:-1]
            lastIndex = indexWithEnd[-1]            
            
        else:
            # Index beibehalten:
            y.index=timeSeriesWithEnd[0:-1]
            lastIndex = timeSeriesWithEnd[-1] # withEnd                
        
        
        def _appendEndIndex(y, lastIndex):   
            # append timestep for full visualization in graphs (steps)
            lastRow = y.iloc[[-1]] # copy last timestep
            lastRow.index = [lastIndex] # replace index

            y =pd.concat([y,lastRow]) # append last row

            return y

        # add last step: (for visu of last timestep-width in plot)
        y = _appendEndIndex(y, lastIndex)                    
    
        return y

    def to_dataFrame(self, busOrComponent:str, direction:str, invert_Output:bool=False)->pd.DataFrame:
        '''
        This Function returns a pd.dataframe containing the OutFlows of the Bus or Comp

        Parameters
        ----------
        busOrComponent : str
            flows linked to this bus or component are chosen
        direction : str ("in","out","inout")
            Direction of the flows to look at. Choose one of "in","out","inout"
        invert_Output : bool
            Wether the output flows should be inverted or not (multiplied by -1)

        Returns
        ---------
        pd.DataFrame
        '''

        (in_flows, out_flows) = self.getFlowsOf(busOrComponent)

        df = pd.DataFrame(index=self.timeSeries)
        if direction=="in":
            flows=in_flows
            for flow in flows:
                flow: cFlow_post
                flow.label_full
                df[flow.label_full] = flow.results['val']
        elif direction=="out":
            flows=out_flows
            for flow in flows:
                flow: cFlow_post
                flow.label_full
                if invert_Output:
                    df[flow.label_full] = flow.results['val'] * -1
                else:
                    df[flow.label_full] = flow.results['val']
        elif direction=="inout":
            for flow in in_flows:
                flow: cFlow_post
                flow.label_full
                df[flow.label_full] = flow.results['val']
            for flow in out_flows:
                flow: cFlow_post
                flow.label_full
                if invert_Output:
                    df[flow.label_full] = flow.results['val']*-1
                else:
                    df[flow.label_full] = flow.results['val']
        else:
            raise Exception(direction+' is no valid arg for "direction" ')
        return df

                             
# self.results[]
## ideen: kosten-übersicht, JDL, Torten-Diag für Busse