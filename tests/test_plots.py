# -*- coding: utf-8 -*-
"""
Manual test script for plots
"""
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly

from flixOpt import plotting


class TestPlots(unittest.TestCase):
    def setUp(self):
        np.random.seed(72)

    @staticmethod
    def get_sample_data(nr_of_columns: int = 7,
                        time_steps_per_period: int = 24,
                        only_pos_or_neg: bool = True,
                        datetime_index : bool = True,
                        column_prefix: str = ''):
        columns = [f"Region {i + 1}{column_prefix}" for i in range(nr_of_columns)]  # More realistic column labels

        if only_pos_or_neg:
            positive_data = np.abs(np.random.rand(time_steps_per_period, nr_of_columns) * 100)
            negative_data = -np.abs(np.random.rand(time_steps_per_period, nr_of_columns) * 100)
            data = pd.DataFrame(np.concatenate([positive_data, negative_data], axis=1),
                                columns=[f"Region {i + 1}" for i in range(nr_of_columns)] +
                                        [f"Region {i + 1} Negative" for i in range(nr_of_columns)])
        else:
            data = pd.DataFrame(np.random.randn(time_steps_per_period, nr_of_columns) * 50 + 20,
                                columns=columns)  # Random data with both positive and negative values
        if datetime_index:
            data.index = pd.date_range('2023-01-01', periods=time_steps_per_period, freq='h')
        return data

    def test_bar_plots(self):
        data = self.get_sample_data()
        plotly.offline.plot(plotting.with_plotly(data, 'bar'))
        plotting.with_matplotlib(data, 'bar')
        plt.show()

    def test_line_plots(self):
        data = self.get_sample_data()
        plotly.offline.plot(plotting.with_plotly(data, 'line'))
        plotting.with_matplotlib(data, 'line')
        plt.show()


if __name__ == '__main__':
    unittest.main()