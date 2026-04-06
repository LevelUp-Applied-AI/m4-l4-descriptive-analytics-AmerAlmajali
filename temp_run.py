
import pandas as pd
import numpy as np
from eda_report import EDAReport

df = pd.DataFrame({
    "a": np.random.randn(100),
    "b": np.random.randn(100),
    "c": np.random.choice(["x", "y"], size=100)
})

report = EDAReport(df, output_dir="test_output")
report.generate_full_report()
