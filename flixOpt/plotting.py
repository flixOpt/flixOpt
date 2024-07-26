# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:43:09 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import matplotlib.pyplot as plt
import pandas as pd


def plotStackedSteps(ax, df: pd.DataFrame, showLegend=True, colors=None):  # df = dataframes!
    # Händische Lösung für stacked steps, da folgendes nicht funktioniert:
    # -->  plt.plot(y_pos.index, y_pos.values, drawstyle='steps-post', stacked=True) -> error
    # -->  ax.stackplot(x, y_pos, labels = labels, drawstyle='steps-post') -> error

    # Aufteilen in positiven und negativen Teil:
    y_pos = df.clip(lower=0)  # postive Werte
    y_neg = df.clip(upper=0)  # negative Werte

    # Stapelwerte:
    y_pos_cum = y_pos.cumsum(axis=1)
    y_neg_cum = y_neg.cumsum(axis=1)

    # plot-funktion
    def plot_y_cum(ax, y_cum, colors, plotLabels=True):
        first = True
        for i in range(len(y_cum.columns)):
            col = y_cum.columns[i]
            y1 = y_cum[col]
            y2 = y_cum[col] * 0 if first else y_cum[y_cum.columns[i - 1]]
            col = y_cum.columns[i]
            label = col if plotLabels else None
            ax.fill_between(x=y_cum.index, y1=y1, y2=y2, label=label, color=colors[i], alpha=1, step='post',
                            linewidth=0)
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

    # plotting:
    plot_y_cum(ax, y_pos_cum, colors, plotLabels=True)
    plot_y_cum(ax, y_neg_cum, colors, plotLabels=False)

    if showLegend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plotFlow(calc, aFlow_value, label, withPoints=True):
    # Linie:
    plt.step(calc.time_series, aFlow_value, where='post', label=label)
    # Punkte dazu:
    if withPoints:
        # TODO: gleiche Farbe!
        # aStr = 'C' + str(i) + 'o'
        aStr = 'o'
        plt.plot(calc.time_series, aFlow_value, aStr)


# TODO: könnte man ggf. schöner mit dict-result machen bzw. sollte auch mit dict-result gehen!
def plotOn(mb, aVar_struct, var_label, y, plotSwitchOnOff=True):
    try:
        plt.step(mb.time_series, aVar_struct.on_ * y, ':', where='post', label=var_label + '_On')
        if plotSwitchOnOff:
            try:
                plt.step(mb.time_series, aVar_struct.switchOn_ * y, '+', where='post', label=var_label + '_SwitchOn')
                plt.step(mb.time_series, aVar_struct.switchOff_ * y, 'x', where='post', label=var_label + '_SwitchOff')
            except:
                pass
    except:
        pass

# # Input z.B. 'results_struct.KWK.Q_th.on' oder [KWK,'Q_th','on']
# def plotSegmentedValue(calc : Calculation, results_struct_As_String_OR_keyList):
#   if length(calc.segmented_system_models) == 0 :
#     raise Exception 'Keine Segmente vorhanden!'
#   else :
#     for aModBox in calc.segmented_system_models:
#       # aVal:
#       eval('aVal = aModBox.' + results_struct_As_String)
#       # aTimeSeries:
#       aTimeSeries = aModBox.time_series_with_end[:length(aVal)] # ggf. um 1 kürzen, wenn kein Speicherladezustand
#       plt.step(aTimeSeries, aVal, where = 'post')
