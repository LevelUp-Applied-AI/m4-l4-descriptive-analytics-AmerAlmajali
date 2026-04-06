import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower


df = pd.read_csv("data/student_performance.csv")

intern = df[df["has_internship"] == "Yes"]["gpa"].dropna().values
no_intern = df[df["has_internship"] == "No"]["gpa"].dropna().values


def bootstrap_ci(data, n_bootstrap=10000):
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)

    return float(lower), float(upper)


ci_intern = bootstrap_ci(intern)
ci_no_intern = bootstrap_ci(no_intern)

print("Bootstrap CI (Intern):", ci_intern)
print("Bootstrap CI (No Intern):", ci_no_intern)


def t_confidence_interval(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf(0.975, len(data) - 1)
    return mean - h, mean + h


print("T-test CI (Intern):", t_confidence_interval(intern))
print("T-test CI (No Intern):", t_confidence_interval(no_intern))


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std


d = cohens_d(intern, no_intern)
print("Effect size (d):", d)


analysis = TTestIndPower()

sample_size = analysis.solve_power(effect_size=d, power=0.8, alpha=0.05)

print("Required sample size per group:", int(sample_size))


def simulate_false_positive(n_sim=1000, sample_size=50):
    false_positives = 0

    for _ in range(n_sim):
        # Same distribution → no real difference
        g1 = np.random.normal(3.0, 0.5, sample_size)
        g2 = np.random.normal(3.0, 0.5, sample_size)

        _, p = stats.ttest_ind(g1, g2)

        if p < 0.05:
            false_positives += 1

    return false_positives / n_sim


fp_rate = simulate_false_positive()
print("False positive rate:", fp_rate)
