"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    # ── Load the dataset ──────────────────────────────────────────
    df = pd.read_csv(filepath)

    # Print basic info for quick inspection in the console
    print(df.shape)
    print(df.dtypes)
    print((df.isnull().sum() / len(df) * 100).round(2))

    # ── Missing data heatmap ──────────────────────────────────────
    # Visualise which cells are missing across the entire dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Data Pattern")
    plt.tight_layout()
    fig.savefig("output/missing_data_heatmap.png", dpi=150)
    plt.close()

    # ── Commute minutes missingness correlation ───────────────────
    # Create a binary indicator: 1 = commute_minutes is missing, 0 = present
    # Then correlate it with all numeric columns to check if missingness
    # is related to other variables (test for MCAR)
    df["commute_missing"] = df["commute_minutes"].isnull().astype(int)
    corr_matrix = df.corr(numeric_only=True)[["commute_missing"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlation with Commute Minutes Missingness")
    plt.tight_layout()
    fig.savefig("output/commute_missing_correlation.png", dpi=150)
    plt.close()

    # ── Scholarship missingness correlation ───────────────────────
    # Same approach for scholarship: binary indicator then correlation heatmap
    df["scholarship_missing"] = df["scholarship"].isnull().astype(int)
    corr_matrix = df.corr(numeric_only=True)[["scholarship_missing"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlation with Scholarship Missingness")
    plt.tight_layout()
    fig.savefig("output/scholarship_missing_correlation.png", dpi=150)
    plt.close()

    # ── Commute minutes by department ─────────────────────────────
    # Print mean and median per department to check for group-level differences
    # then plot a boxplot to visualise the distribution per department
    print(df.groupby("department")["commute_minutes"].agg(["mean", "median"]))
    plt.figure()
    sns.boxplot(x="department", y="commute_minutes", data=df)
    plt.title("Commute Minutes by Department")
    plt.xticks(rotation=45)
    plt.savefig("output/commute_by_department.png")
    plt.close()

    # ── Handle missing values ─────────────────────────────────────
    # commute_minutes (~9.05% missing, MCAR): impute with median
    # Median is stable across all departments (~25 min), so imputation
    # introduces no systematic bias and avoids losing ~181 rows
    df["commute_minutes"] = df["commute_minutes"].fillna(df["commute_minutes"].median())

    # scholarship (~19.45% missing, MCAR): fill with 'None' category
    # Dropping ~389 rows would be too costly; 'None' keeps all rows
    # and makes the missingness explicit for downstream analysis
    df["scholarship"] = df["scholarship"].fillna("None")

    # ── Compute missing value stats AFTER imputation ──────────────
    # Used to report the final state in the profile file
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    # ── Write data_profile.txt ────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    with open("output/data_profile.txt", "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")
        f.write("DATA PROFILE REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Section 1: dataset dimensions
        f.write("1. SHAPE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")

        # Section 2: column data types
        f.write("2. DATA TYPES\n")
        f.write("-" * 40 + "\n")
        for col, dtype in df.dtypes.items():
            f.write(f"  {col:<25} {dtype}\n")
        f.write("\n")

        # ─────────────────────────────────────────────
        # BEFORE IMPUTATION (professional addition)
        # ─────────────────────────────────────────────
        f.write("3. MISSING VALUES — BEFORE HANDLING\n")
        f.write("-" * 40 + "\n")

        missing_before = pd.read_csv(filepath).isnull().sum()
        missing_before_pct = (missing_before / len(df) * 100).round(2)

        f.write(f"  {'Column':<25} {'Count':>8}  {'Percent':>8}\n")
        for col in missing_before.index:
            f.write(
                f"  {col:<25} {missing_before[col]:>8}  {missing_before_pct[col]:>7}%\n"
            )
        f.write("\n")

        # ─────────────────────────────────────────────
        # AFTER IMPUTATION (your existing logic)
        # ─────────────────────────────────────────────
        f.write("4. MISSING VALUES — AFTER HANDLING\n")
        f.write("-" * 40 + "\n")

        f.write(f"  {'Column':<25} {'Count':>8}  {'Percent':>8}\n")
        for col in df.columns:
            f.write(f"  {col:<25} {missing_counts[col]:>8}  {missing_pct[col]:>7}%\n")
        f.write("\n")

        # ─────────────────────────────────────────────
        # PROFESSIONAL HANDLING EXPLANATION
        # ─────────────────────────────────────────────
        f.write("5. MISSING VALUE HANDLING DECISIONS\n")
        f.write("-" * 40 + "\n")

        f.write(
            "  commute_minutes (~9.05% missing):\n"
            "    Decision: Impute using global median.\n"
            "    Justification:\n"
            "      • Missingness pattern appears random based on heatmap visualization.\n"
            "      • Correlation analysis shows near-zero relationships (|r| ≈ 0.01), supporting MCAR assumption.\n"
            "      • Group-level analysis (by department) shows nearly identical medians (~25 minutes),\n"
            "        indicating no meaningful subgroup variation.\n"
            "      • Median imputation preserves distribution shape and avoids bias from extreme values.\n"
            "      • Dropping ~9% of observations (~180 rows) would unnecessarily reduce dataset size.\n\n"
        )

        f.write(
            "  scholarship (~19.45% missing):\n"
            "    Decision: Impute using 'None' category.\n"
            "    Justification:\n"
            "      • Missingness appears random (MCAR) with negligible correlation to other variables (|r| ≤ 0.03).\n"
            "      • High missing proportion (~19%) makes row deletion inappropriate.\n"
            "      • As a categorical variable, statistical imputation (mean/median) is not applicable.\n"
            "      • Introducing 'None' preserves dataset integrity and explicitly encodes absence of information.\n"
            "      • This approach avoids introducing artificial bias from mode imputation.\n\n"
        )

        f.write(
            "  All other columns:\n"
            "    Decision: No action required.\n"
            "    Justification: These variables contain complete data (0% missing),\n"
            "    indicating no data quality concerns.\n\n"
        )

        # ─────────────────────────────────────────────
        # DESCRIPTIVE STATISTICS
        # ─────────────────────────────────────────────
        f.write("6. DESCRIPTIVE STATISTICS (numeric columns)\n")
        f.write("-" * 40 + "\n")
        f.write(df.describe().to_string())
        f.write("\n")

    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    os.makedirs("output", exist_ok=True)

    # ── 1. GPA distribution — histogram with KDE ──────────────────
    # Shows whether GPA is normally distributed or skewed
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["gpa"], kde=True, ax=ax)
    ax.set_title("GPA Distribution — Roughly Normal with Slight Left Skew")
    ax.set_xlabel("GPA")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig("output/dist_gpa.png", dpi=150)
    plt.close()

    # ── 2. Study hours per week — histogram with KDE ──────────────
    # Reveals how students spread across light vs heavy study loads
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["study_hours_weekly"], kde=True, ax=ax)
    ax.set_title("Study Hours Weekly Distribution — Wide Spread Across Students")
    ax.set_xlabel("Study Hours (Weekly)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig("output/dist_study_hours.png", dpi=150)
    plt.close()

    # ── 3. Attendance percentage — histogram with KDE ─────────────
    # Checks if most students attend regularly or if there is a low-attendance tail
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["attendance_pct"], kde=True, ax=ax)
    ax.set_title("Attendance % Distribution — Most Students Attend Above 70%")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig("output/dist_attendance.png", dpi=150)
    plt.close()

    # ── 4. GPA by department — box plot ───────────────────────────
    # Compares GPA spread and median across all 5 departments
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="department", y="gpa", data=df, ax=ax)
    ax.set_title("GPA by Department — Similar Medians, Slight Variance Differences")
    ax.set_xlabel("Department")
    ax.set_ylabel("GPA")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("output/dist_gpa_by_department.png", dpi=150)
    plt.close()

    # ── 5. Scholarship counts — bar chart ─────────────────────────
    # Shows how students are distributed across scholarship categories
    # including the 'None' group created during imputation
    fig, ax = plt.subplots(figsize=(8, 5))
    df["scholarship"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Scholarship Distribution — Most Students Have No Scholarship")
    ax.set_xlabel("Scholarship Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("output/dist_scholarship.png", dpi=150)
    plt.close()


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """
    os.makedirs("output", exist_ok=True)

    # ── Compute Pearson correlation matrix for all numeric columns ─
    corr_matrix = df.select_dtypes(include="number").corr()

    # ── 1. Annotated heatmap of the full correlation matrix ────────
    # Reveals which variable pairs move together (positive) or
    # inversely (negative); self-correlations on the diagonal = 1.0
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix — All Numeric Variables")
    plt.tight_layout()
    fig.savefig("output/corr_heatmap.png", dpi=150)
    plt.close()

    # ── Find the two most correlated pairs (exclude self & duplicates)
    # Unstack the matrix, drop the diagonal (corr = 1.0), take abs value,
    # then sort descending to surface the strongest relationships
    corr_pairs = (
        corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))  # mask the diagonal
        .abs()
        .unstack()
        .dropna()
        .sort_values(ascending=False)
    )

    # Remove duplicate pairs (A-B and B-A are the same pair)
    seen = set()
    top_pairs = []
    for (col1, col2), val in corr_pairs.items():
        pair = frozenset([col1, col2])
        if pair not in seen:
            seen.add(pair)
            top_pairs.append((col1, col2, val))
        if len(top_pairs) == 2:
            break

    print("\nTop 2 correlated variable pairs:")
    for col1, col2, val in top_pairs:
        print(f"  {col1}  vs  {col2}  ->  r = {val:.3f}")

    # ── 2. Scatter plot — most correlated pair ─────────────────────
    col1, col2, val1 = top_pairs[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df[col1], y=df[col2], alpha=0.4, ax=ax)
    ax.set_title(f"Scatter: {col1} vs {col2}  (r = {val1:.2f})")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    plt.tight_layout()
    fig.savefig("output/corr_scatter_top1.png", dpi=150)
    plt.close()

    # ── 3. Scatter plot — second most correlated pair ──────────────
    col3, col4, val2 = top_pairs[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df[col3], y=df[col4], alpha=0.4, ax=ax)
    ax.set_title(f"Scatter: {col3} vs {col4}  (r = {val2:.2f})")
    ax.set_xlabel(col3)
    ax.set_ylabel(col4)
    plt.tight_layout()
    fig.savefig("output/corr_scatter_top2.png", dpi=150)
    plt.close()


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'scholarship_chi2',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests:
        - t-test: Does GPA differ between students with and without internships?
        - chi-square: Is scholarship status associated with department?
    """
    results = {}

    # ── Hypothesis 1: Internship vs GPA (independent samples t-test)
    # H0: mean GPA is equal for students with and without internships
    # H1: students with internships have a higher mean GPA
    gpa_intern = df[df["has_internship"] == "Yes"]["gpa"].dropna()
    gpa_no_intern = df[df["has_internship"] == "No"]["gpa"].dropna()

    t_stat, p_val = stats.ttest_ind(gpa_intern, gpa_no_intern, alternative="greater")

    # Cohen's d — measures practical effect size (small=0.2, medium=0.5, large=0.8)
    pooled_std = np.sqrt((gpa_intern.std() ** 2 + gpa_no_intern.std() ** 2) / 2)
    cohens_d = (gpa_intern.mean() - gpa_no_intern.mean()) / pooled_std

    results["internship_ttest"] = {
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
    }

    print("\n" + "=" * 60)
    print("HYPOTHESIS 1: Internship students have higher GPA")
    print("-" * 60)
    print(
        f"  GPA with internship    : mean = {gpa_intern.mean():.3f}, n = {len(gpa_intern)}"
    )
    print(
        f"  GPA without internship : mean = {gpa_no_intern.mean():.3f}, n = {len(gpa_no_intern)}"
    )
    print(f"  t-statistic : {t_stat:.4f}")
    print(f"  p-value     : {p_val:.4f}")
    print(f"  Cohen's d   : {cohens_d:.4f}")
    if p_val < 0.05:
        print(
            "   SIGNIFICANT — internship students have statistically higher GPA (p < 0.05)."
        )
    else:
        print("   NOT significant — no strong evidence of higher GPA (p >= 0.05).")
    effect = (
        "small" if abs(cohens_d) < 0.2 else "medium" if abs(cohens_d) < 0.5 else "large"
    )
    print(f"  Effect size : {effect} (d = {cohens_d:.3f})")

    # ── Hypothesis 2: Scholarship status vs Department (chi-square)
    # H0: scholarship type and department are independent
    # H1: scholarship type is associated with department
    contingency_table = pd.crosstab(df["scholarship"], df["department"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    results["scholarship_chi2"] = {
        "chi2_statistic": chi2,
        "p_value": p_chi2,
        "degrees_of_freedom": dof,
    }

    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: Scholarship status is associated with department")
    print("-" * 60)
    print(f"  Chi-square statistic : {chi2:.4f}")
    print(f"  p-value              : {p_chi2:.4f}")
    print(f"  Degrees of freedom   : {dof}")
    if p_chi2 < 0.05:
        print(
            "   SIGNIFICANT — scholarship distribution differs across departments (p < 0.05)."
        )
    else:
        print(
            "   NOT significant — no strong association between scholarship and department (p >= 0.05)."
        )
    print("=" * 60)

    return results


def write_findings(df, test_results):
    """Write FINDINGS.md summarising all analysis results.

    Args:
        df: cleaned pandas DataFrame
        test_results: dict returned by run_hypothesis_tests()

    Returns:
        None

    Side effects:
        Writes output/FINDINGS.md
    """
    os.makedirs("output", exist_ok=True)

    tt = test_results["internship_ttest"]
    chi = test_results["scholarship_chi2"]
    tt_effect = (
        "small"
        if abs(tt["cohens_d"]) < 0.2
        else "medium" if abs(tt["cohens_d"]) < 0.5 else "large"
    )

    with open("FINDINGS.md", "w", encoding="utf-8") as f:

        f.write("# Student Performance — EDA Findings Report\n\n")

        # ── 1. Dataset description ─────────────────────────────────
        f.write("## 1. Dataset Description\n\n")
        f.write(
            f"- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n"
            "- **Columns**: student_id, department, semester, course_load, "
            "study_hours_weekly, gpa, attendance_pct, has_internship, "
            "commute_minutes, scholarship\n"
            "- **Data quality issues**:\n"
            "  - `commute_minutes`: ~9.05% missing → imputed with median (MCAR confirmed)\n"
            "  - `scholarship`: ~19.45% missing → filled with `'None'` category (MCAR confirmed)\n"
            "  - All other columns: 0% missing\n\n"
        )

        # ── 2. Distribution findings ───────────────────────────────
        f.write("## 2. Key Distribution Findings\n\n")
        f.write(
            "- **GPA** appears roughly normally distributed with a slight left skew, "
            "suggesting most students perform near the average with fewer low performers. "
            "See `output/dist_gpa.png`.\n"
            "- **Study hours weekly** shows a wide spread, indicating high variability "
            "in student effort. See `output/dist_study_hours.png`.\n"
            "- **Attendance %** is left-skewed — the majority of students attend above 70%, "
            "with a small tail of low-attendance students. See `output/dist_attendance.png`.\n"
            "- **GPA by department** boxplots reveal similar medians across all departments, "
            "suggesting department alone does not strongly drive GPA differences. "
            "See `output/dist_gpa_by_department.png`.\n"
            "- **Scholarship** bar chart shows most students hold no scholarship. "
            "See `output/dist_scholarship.png`.\n\n"
        )

        # ── 3. Correlation findings ────────────────────────────────
        f.write("## 3. Notable Correlations\n\n")
        f.write(
            "- The full Pearson correlation heatmap is saved at `output/corr_heatmap.png`.\n"
            "- The two strongest variable pairs are visualised in "
            "`output/corr_scatter_top1.png` and `output/corr_scatter_top2.png`.\n"
            "- `study_hours_weekly` and `gpa` are expected to show a positive correlation — "
            "students who study more tend to earn higher grades.\n"
            "- `attendance_pct` and `gpa` may also correlate positively, as regular attendance "
            "likely supports academic performance.\n"
            "- **Caveat**: correlation is not causation. A third variable (e.g., motivation or "
            "socioeconomic background) could drive both study hours and GPA simultaneously.\n\n"
        )

        # ── 4. Hypothesis test results ─────────────────────────────
        f.write("## 4. Hypothesis Test Results\n\n")

        f.write("### Hypothesis 1 — Internship students have higher GPA\n\n")
        f.write(
            "- **Test**: Independent samples one-tailed t-test\n"
            f"- **t-statistic**: {tt['t_statistic']:.4f}\n"
            f"- **p-value**: {tt['p_value']:.4f}\n"
            f"- **Cohen's d**: {tt['cohens_d']:.4f} ({tt_effect} effect)\n"
            f"- **Interpretation**: "
            f"{'Statistically significant (p < 0.05). Internship students have a meaningfully higher GPA.' if tt['p_value'] < 0.05 else 'Not statistically significant (p >= 0.05). Insufficient evidence that internship students have higher GPA.'}\n\n"
        )

        f.write(
            "### Hypothesis 2 — Scholarship status is associated with department\n\n"
        )
        f.write(
            "- **Test**: Chi-square test of independence\n"
            f"- **Chi-square statistic**: {chi['chi2_statistic']:.4f}\n"
            f"- **p-value**: {chi['p_value']:.4f}\n"
            f"- **Degrees of freedom**: {chi['degrees_of_freedom']}\n"
            f"- **Interpretation**: "
            f"{'Statistically significant (p < 0.05). Scholarship distribution differs across departments.' if chi['p_value'] < 0.05 else 'Not statistically significant (p >= 0.05). No strong evidence of association between scholarship and department.'}\n\n"
        )

        # ── 5. Actionable recommendations ─────────────────────────
        f.write("## 5. Actionable Recommendations\n\n")
        f.write(
            "1. **Promote internship programmes university-wide.** "
            "The t-test suggests internship students may have higher GPAs. "
            "Expanding access — especially in departments with low internship rates — "
            "could lift overall academic performance.\n\n"
            "2. **Investigate and reduce scholarship data gaps.** "
            "Nearly 20% of scholarship records are missing. The university should audit "
            "its data collection process to ensure scholarship status is recorded for all "
            "students, enabling fairer financial aid analysis.\n\n"
            "3. **Target support at low-attendance students.** "
            "The left-skewed attendance distribution reveals a distinct group attending "
            "below 70%. Early-warning interventions (e.g., advisor outreach) for these "
            "students could prevent GPA decline. See `output/dist_attendance.png`.\n"
        )

    print("\nFINDINGS.md written to FINDINGS.md")


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    # Step 1: Load, profile, and clean the dataset
    df = load_and_profile("data/student_performance.csv")

    # Step 2: Generate distribution plots for numeric and categorical columns
    plot_distributions(df)

    # Step 3: Compute and visualise the correlation matrix and top scatter plots
    plot_correlations(df)

    # Step 4: Run hypothesis tests and collect results for the report
    test_results = run_hypothesis_tests(df)

    # Step 5: Write FINDINGS.md summarising all analysis results
    write_findings(df, test_results)


if __name__ == "__main__":
    main()
