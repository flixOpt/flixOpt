# -*- coding: utf-8 -*-
"""
Manual test script for plots
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly

from flixOpt import plotting


if __name__ == '__main__':
    only_pos_or_neg = False
    nr_of_periods = 30
    time_steps_per_period = 24
    columns = [f"Region {i+1}" for i in range(nr_of_periods)]  # More realistic column labels

    if only_pos_or_neg:
        positive_data = np.abs(np.random.rand(time_steps_per_period, nr_of_periods) * 100)
        negative_data = -np.abs(np.random.rand(time_steps_per_period, nr_of_periods) * 100)
        data = pd.DataFrame(np.concatenate([positive_data, negative_data], axis=1),
                            columns=[f"Region {i + 1}" for i in range(nr_of_periods)] +
                                    [f"Region {i + 1} Negative" for i in range(nr_of_periods)])
    else:
        data = pd.DataFrame(np.random.randn(time_steps_per_period, nr_of_periods) * 50 + 20, columns=columns)  # Random data with both positive and negative values

    data.index = pd.date_range('2023-01-01', periods=time_steps_per_period, freq='h')

    mode = 'bar'
    # Plot with Plotly
    plotly.offline.plot(plotting.with_plotly(data, mode))

    # Plot with Matplotlib
    plotting.with_matplotlib(data, mode)
    plt.show()

    print()