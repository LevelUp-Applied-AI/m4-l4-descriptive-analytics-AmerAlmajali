# eda_report.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDAReport:
    def __init__(
        self,
        df,
        output_dir="output",
        columns=None,
        style="whitegrid",
        save_plots=True,
        verbose=True,
        run_box_violin=True,
        run_correlation=True,
    ):
        self.df = df.copy()

        # Safe column selection
        self.columns = [
            col for col in (columns if columns else df.columns) if col in df.columns
        ]

        self.numeric_cols = (
            self.df[self.columns].select_dtypes(include=np.number).columns
        )

        self.output_dir = output_dir
        self.save_plots = save_plots
        self.verbose = verbose
        self.run_box_violin = run_box_violin
        self.run_correlation = run_correlation

        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style(style)

    # ─────────────────────────────────────────────
    # Logging helper
    # ─────────────────────────────────────────────
    def _log(self, message):
        if self.verbose:
            print(f"[EDA] {message}")

    # ─────────────────────────────────────────────
    # 1. DATA PROFILE
    # ─────────────────────────────────────────────
    def data_profile(self):
        self._log("Generating data profile...")

        profile = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "missing_count": self.df.isnull().sum(),
                "missing_%": (self.df.isnull().sum() / len(self.df) * 100).round(2),
                "unique_values": self.df.nunique(),
            }
        )

        path = f"{self.output_dir}/data_profile.csv"
        profile.to_csv(path)

        self._log(f"Saved → {path}")
        return profile

    # ─────────────────────────────────────────────
    # 2. DISTRIBUTIONS
    # ─────────────────────────────────────────────
    def plot_distributions(self):
        self._log("Plotting distributions...")

        for col in self.numeric_cols:
            try:
                fig, ax = plt.subplots()

                sns.histplot(self.df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"{col} Distribution")

                if self.save_plots:
                    path = f"{self.output_dir}/dist_{col}.png"
                    fig.savefig(path)
                    self._log(f"Saved → {path}")

                plt.close(fig)

            except Exception as e:
                self._log(f"Skipping {col}: {e}")

    # ─────────────────────────────────────────────
    # 3. BOX + VIOLIN
    # ─────────────────────────────────────────────
    def plot_box_violin(self):
        self._log("Plotting boxplots & violin plots...")

        for col in self.numeric_cols:
            try:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))

                sns.boxplot(y=self.df[col], ax=ax[0])
                ax[0].set_title(f"{col} Boxplot")

                sns.violinplot(y=self.df[col], ax=ax[1])
                ax[1].set_title(f"{col} Violin")

                if self.save_plots:
                    path = f"{self.output_dir}/box_violin_{col}.png"
                    fig.savefig(path)
                    self._log(f"Saved → {path}")

                plt.close(fig)

            except Exception as e:
                self._log(f"Skipping {col}: {e}")

    # ─────────────────────────────────────────────
    # 4. CORRELATION
    # ─────────────────────────────────────────────
    def correlation_analysis(self):
        self._log("Analyzing correlations...")

        if len(self.numeric_cols) < 2:
            self._log("Not enough numeric columns.")
            return None

        corr = self.df[self.numeric_cols].corr()

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)

        path = f"{self.output_dir}/correlation_heatmap.png"
        fig.savefig(path)
        plt.close(fig)

        self._log(f"Saved → {path}")

        # Remove duplicate pairs
        corr_pairs = corr.where(~np.eye(len(corr), dtype=bool)).abs().unstack().dropna()

        corr_pairs = corr_pairs[corr_pairs.index.map(lambda x: x[0] < x[1])]
        corr_pairs = corr_pairs.sort_values(ascending=False)

        top_pairs = corr_pairs.head(5)
        top_pairs.to_csv(f"{self.output_dir}/top_correlations.csv")

        self._log("Saved top correlations")

        return corr

    # ─────────────────────────────────────────────
    # 5. MISSING DATA
    # ─────────────────────────────────────────────
    def missing_analysis(self):
        self._log("Analyzing missing data...")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, ax=ax)

        path = f"{self.output_dir}/missing_data.png"
        fig.savefig(path)
        plt.close(fig)

        # Save numeric summary
        missing_summary = self.df.isnull().sum()
        missing_summary.to_csv(f"{self.output_dir}/missing_summary.csv")

        self._log(f"Saved → {path}")

    # ─────────────────────────────────────────────
    # 6. OUTLIERS (IQR)
    # ─────────────────────────────────────────────
    def outlier_summary(self):
        self._log("Detecting outliers...")

        results = []

        for col in self.numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)][col]

            results.append(
                {
                    "column": col,
                    "outlier_count": len(outliers),
                    "outlier_%": (len(outliers) / len(self.df)) * 100,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "min": self.df[col].min(),
                    "max": self.df[col].max(),
                    "mean": self.df[col].mean(),
                }
            )

        out_df = pd.DataFrame(results)

        path = f"{self.output_dir}/outliers_detailed.csv"
        out_df.to_csv(path, index=False)

        self._log(f"Saved → {path}")

        return out_df

    # ─────────────────────────────────────────────
    # MASTER FUNCTION
    # ─────────────────────────────────────────────
    def generate_full_report(self):
        self._log("Starting full EDA report...")

        self.data_profile()
        self.plot_distributions()

        if self.run_box_violin:
            self.plot_box_violin()

        if self.run_correlation:
            self.correlation_analysis()

        self.missing_analysis()
        self.outlier_summary()

        self._log("EDA report completed successfully!")
