from __future__ import division
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def load_race_income_dc():
    race_income = pd.read_pickle("./data/race_income_dc.npy")
    binary_incomes = race_income[
        (race_income["RAC1P"] == 2) | (race_income["RAC1P"] == 1)
    ].dropna(axis=0)
    num_white_records = binary_incomes[binary_incomes["RAC1P"] == 1].shape[0]
    print(
        "Loaded {} records, with {} white and {} black".format(
            binary_incomes.shape[0],
            num_white_records,
            binary_incomes.shape[0] - num_white_records,
        )
    )
    race_income = binary_incomes
    return race_income


def compute_KDEs(race_income_df, bandwidths):
    """Compute and return Kernel Density Estimates of the densities
    of the incomes given race"""
    bandwidth1, bandwidth2 = bandwidths
    white_incomes = race_income_df["HINCP"][race_income_df["RAC1P"] == 1]
    white_incomes_array = np.array(white_incomes.dropna())
    white_KD = KernelDensity(bandwidth=bandwidth1)
    white_KD.fit(white_incomes_array.reshape(-1, 1))

    black_incomes = race_income_df["HINCP"][race_income_df["RAC1P"] == 2]
    black_incomes_array = np.array(black_incomes.dropna())
    black_KD = KernelDensity(bandwidth=bandwidth2)
    black_KD.fit(black_incomes_array.reshape(-1, 1))
    u0_frac = float(np.sum(race_income_df["RAC1P"] == 2)) / len(race_income_df)

    return white_KD, black_KD, u0_frac
