# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:13:31 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

# TODO: 
  # -> getVector() -> int32 Vektoren möglich machen

import numpy as np
import re
import math # für nan
import matplotlib.pyplot as plt

from .flixBasicsPublic import cTSraw


def getVector(aValue,aLen):  
  '''
  Macht aus Skalar einen Vektor. Vektor bleibt Vektor. 
  -> Idee dahinter: Aufruf aus abgespeichertem Vektor schneller, als für jede i-te Gleichung zu Checken ob Vektor oder Skalar)
  
  Parameters
  ----------
  
  aValue: skalar, list, np.array
  aLen  : skalar
  '''
  # dtype = 'float64' # -> muss mit übergeben werden, sonst entstehen evtl. int32 Reihen (dort ist kein +/-inf möglich)
  
  # Wenn Skalar oder None
  if (np.isscalar(aValue)) or (aValue is None):             
    aValue = [aValue] * aLen # Liste erstellen
  # Wenn Vektor 
  elif len(aValue)==aLen: 
    pass
  # Wenn Vektor nicht richtige Länge
  else :
    raise Exception('error in changing to len = ' + str(aLen) + '; vector has already len = ' + str(len(aValue)))
  
  aVector = np.array(aValue) # np-array erstellen, wenn es nicht schon einer ist.
  return aVector

# Nimmt Subset der Indexe, wenn es ein Vektor ist. Skalar bleibt skalar.
#   Falls aIndexe = None, dann bleibt es wie es ist.
def getSubSetIfVector(aValue, aIndexe):
  # Wenn Skalar oder None:
  if (np.isscalar(aValue)) or (aValue is None):   
    pass # do nothing
  # wenn aIndexe = None:
  elif aIndexe is None:
    pass # do nothing
  # wenn Vektor:
  else:
    if max(aIndexe) > len(aValue)-1 :
      raise Exception('subSet Indices sind höher als len(Ausgangsvektor)')
    else :
      aValue = aValue[aIndexe]
  return aValue      

# changes zeros to Nans in Vector:
def zerosToNans(aVector):
  # nanVector                 = aVector.copy()
  nanVector                 = aVector.astype(float) # Binär ist erstmal int8-Vektor, Nan gibt es da aber nicht
  nanVector[nanVector == 0] = math.nan
  return nanVector
  
def checkBoundsOfParameter(aParam,aParamLabel,aBounds,aObject=None):
  if  isinstance(aParam, cTSraw):
    aParam=aParam.value
  if np.any(aParam < aBounds[0]) | np.any(aParam >= aBounds[1]):
    raise Exception(aParamLabel + ' verletzt min/max - Grenzen!')
    
# löscht alle in Attributen ungültigen Zeichen: todo: Vollständiger machen!
def checkForAttributeNameConformity(aName):
  char_map = {ord('ä'):'ae',
              ord('ü'):'ue',
              ord('ö'):'oe',
              ord('ß'):'ss',
              ord('-'):'_'}
  newName = aName.translate(char_map)
  if newName != aName:
    print('Name \'' + aName + '\' ist nicht Attributnamen-konform und wird zu \'' + newName + '\' geändert' )
  
  # check, ob jetzt valid variable name: (für Verwendung in results_struct notwendig)
  import re
  if not re.search(r'^[a-zA-Z_]\w*$',newName):
    raise Exception('label \'' + aName + '\' is not valid for variable name \n .\
                     (no number first, no special characteres etc.)')
  return newName
  

class InfiniteFullSet(object):
    def __and__(self, item): # left operand of &
        return item

    def __rand__(self,item): # right operand of &
        return item

    def __str__(self):
        return ('<InfiniteFullSet>')

def plotStackedSteps(ax,df,showLegend=True, colors = None): # df = dataframes!
# Händische Lösung für stacked steps, da folgendes nicht funktioniert:
     # -->  plt.plot(y_pos.index, y_pos.values, drawstyle='steps-post', stacked=True) -> error
     # -->  ax.stackplot(x, y_pos, labels = labels, drawstyle='steps-post') -> error

  # Aufteilen in positiven und negativen Teil: 
  y_pos = df.clip(lower=0) # postive Werte
  y_neg = df.clip(upper=0) # negative Werte
        
  # Stapelwerte:
  y_pos_cum = y_pos.cumsum(axis=1)
  y_neg_cum = y_neg.cumsum(axis=1)
  
  # plot-funktion  
  def plot_y_cum(ax, y_cum, colors, plotLabels = True):    
    first = True
    for i in range(len(y_cum.columns)):
      col = y_cum.columns[i]
      y1 = y_cum[col]    
      y2 = y_cum[col]*0 if first else y_cum[y_cum.columns[i-1]]
      col = y_cum.columns[i]    
      label = col if plotLabels else None
      ax.fill_between(x  = y_cum.index, y1 = y1, y2 = y2, label=label, color = colors[i], alpha=1, step='post', linewidth=0)
      first = False  
  
  
  # colorlist -> damit gleiche Farben für pos und neg - Werte!:
 
  if colors is None:
    colors = []
    # ersten 10 Farben:
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # weitere Farben:
    import matplotlib.colors as mpl_col
    moreColors = mpl_col.cnames.values()
    # anhängen:
    colors += moreColors
  
  
  #plotting:
  plot_y_cum(ax, y_pos_cum, colors, plotLabels = True)
  plot_y_cum(ax, y_neg_cum, colors, plotLabels = False)    

  if showLegend :
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
     

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False  


# Macht aus verschachteltem Dict ein "matlab-struct"-like object
# --> als FeldNamen wird key verwendet wenn string, sonst key.label

# --> dict[key1]['key1_1'] -> struct.key1.key1_1
def createStructFromDictInDict(aDict): 
  # z.B.:
  #      {Kessel_object : {'Q_th':{'on':[..], 'val': [..]}
  #                        'Pel' :{'on':[..], 'val': [..]} } 
  #       Last_object   : {'Q_th':{'val': [..]           } } }

  aStruct = cDataBox2()
  if isinstance(aDict, dict) :
    for key, val in aDict.items() :
      
      ## 1. attr-name :
      
      # Wenn str (z.b. 'Q_th') :
      if isinstance(key, str):
        name = key
      # sonst (z.B. bei Kessel_object):
      else:
        try :
          name = key.label
        except: 
          raise Exception('key has no label!')
      
      ## 2. value:       
      # Wenn Wert wiederum dict, dann rekursiver Aufruf:
      if isinstance(val, dict):
        value = createStructFromDictInDict(val) # rekursiver Aufruf!
      else:
        value = val
      
      if hasattr(aStruct, name) :
        name = name + '_2'      
      setattr(aStruct, name, value)
  else:
    raise Exception('fct needs dict!')
  
  return aStruct

# emuliert matlab - struct-datentyp. (-> durch hinzufügen von Attributen)
class cDataBox2 :
  # pass
  def __str__(self):
    astr =  ('<cDataBox with ' +str( len(self.__dict__)) + ' values: ' )
    for aAttr in self.__dict__.keys():
      astr += aAttr + ', '
    astr += '>'
    return astr
   

def getTimeSeriesWithEnd(timeSeries, dt_last = None):

  ######################################
  ## letzten Zeitpunkt hinzufügen:
  # Wenn nicht gegeben:
  if dt_last is None :
    # wie vorletztes dt:
    dt_last      = timeSeries[-1] - timeSeries[-2]
    
  # Zeitpunkt nach letztem Zeitschritt:
  t_end       = timeSeries[-1] + dt_last  
  timeSeriesWithEnd = np.append(timeSeries, t_end)
  
  return timeSeriesWithEnd
  
''' Tests:
import flixOptHelperFcts as helpers

helpers.getVector(3,10)
helpers.getVector(np.array([1,2]),3)


'''

# check sowohl für globale Zeitreihe, als auch für chosenIndexe:
def checkTimeSeries(aStr, timeSeries):
  
  # Zeitdifferenz:
  #              zweites bis Letztes            - erstes bis Vorletztes
  dt           = timeSeries[1:] - timeSeries[0:-1]
  # dtInHours    = dt.total_seconds() / 3600   
  dtInHours    = dt/np.timedelta64(1, 'h')
  
  # unterschiedliche dt:
  if max(dtInHours)-min(dtInHours) != 0:
    print(aStr + ': !! Achtung !! unterschiedliche delta_t von ' + str(min(dt)) + 'h bis ' + str(max(dt)) + ' h')
  # negative dt:
  if min(dtInHours)<0 :
    raise Exception(aStr + ': Zeitreihe besitzt Zurücksprünge - vermutlich Zeitumstellung nicht beseitigt!')
  
import yaml  
def printDictAndList(aDictOrList):
  print(yaml.dump(aDictOrList, 
                  default_flow_style = False,
                  width = 1000, # verhindern von zusätzlichen Zeilenumbrüchen
                  allow_unicode = True))

  # # dict:
  # if isinstance(aDictOrList, dict):
  #   for key, value in aDictOrList.items():
      
  #     # wert ist ...
  #     # 1. dict/liste:
  #     if isinstance(value,dict) | isinstance(value,list):
  #       print(place + str(key) + ': ')
  #       printDictAndList(value, place + '  ')                        
  #     # 2. None:
  #     elif value == [None]:
  #       print(place + str(key) + ': ' + str(value))            
  #     # 3. wert:
  #     else: 
  #       print(place + str(key) + ': ' + str(value))      
  # # liste:
  # else:
  #   for i in range(len(aDictOrList)):        
  #     print('' + str(i) + ': ')
  #     printDictAndList(aDictOrList[i], place + '  ')

  

# max from num-lists and skalars
# arg = list, array, skalar
# example: max_of_lists_and_scalars([1,2],3) --> 3
def max_args(*args):
  array = _mergeToArray(args)
  return array.max()

# example: min_of_lists_and_scalars([1,2],3) --> 1
def min_args(*args):
  array = _mergeToArray(args)
  return array.min()

def _mergeToArray(args):
  array = np.array([])
  for i in range(len(args)):
    
    if np.isscalar(args[i]):       
      arg = [args[i]] # liste draus machen      
    else:
      arg = args[i]
    array = np.append(array, arg)           
  return array