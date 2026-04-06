# test_eda_report.py

import pandas as pd
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eda_report import EDAReport


def test_random_data():
    df = pd.DataFrame(
        {"a": np.random.randn(100), "b": np.random.randn(100), "c": ["x"] * 100}
    )
    report = EDAReport(df, output_dir="test1")
    report.generate_full_report()
    assert os.path.exists("test1/data_profile.csv"), "Profile not saved"
    assert os.path.exists("test1/correlation_heatmap.png"), "Heatmap not saved"
    assert os.path.exists("test1/missing_data.png"), "Missing data plot not saved"


def test_missing_values():
    df = pd.DataFrame({"x": [1, np.nan, 3, np.nan], "y": [np.nan, 2, 3, 4]})

    report = EDAReport(df, output_dir="test2")
    report.generate_full_report()


def test_small_df():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    report = EDAReport(df, output_dir="test3")
    report.generate_full_report()
