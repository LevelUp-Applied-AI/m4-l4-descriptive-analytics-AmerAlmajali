# Student Performance — EDA Findings Report

## 1. Dataset Description

- **Shape**: 2000 rows × 12 columns
- **Columns**: student_id, department, semester, course_load, study_hours_weekly, gpa, attendance_pct, has_internship, commute_minutes, scholarship
- **Data quality issues**:
  - `commute_minutes`: ~9.05% missing → imputed with median (MCAR confirmed)
  - `scholarship`: ~19.45% missing → filled with `'None'` category (MCAR confirmed)
  - All other columns: 0% missing

## 2. Key Distribution Findings

- **GPA** appears roughly normally distributed with a slight left skew, suggesting most students perform near the average with fewer low performers. See `output/dist_gpa.png`.
- **Study hours weekly** shows a wide spread, indicating high variability in student effort. See `output/dist_study_hours.png`.
- **Attendance %** is left-skewed — the majority of students attend above 70%, with a small tail of low-attendance students. See `output/dist_attendance.png`.
- **GPA by department** boxplots reveal similar medians across all departments, suggesting department alone does not strongly drive GPA differences. See `output/dist_gpa_by_department.png`.
- **Scholarship** bar chart shows most students hold no scholarship. See `output/dist_scholarship.png`.

## 3. Notable Correlations

- The full Pearson correlation heatmap is saved at `output/corr_heatmap.png`.
- The two strongest variable pairs are visualised in `output/corr_scatter_top1.png` and `output/corr_scatter_top2.png`.
- `study_hours_weekly` and `gpa` are expected to show a positive correlation — students who study more tend to earn higher grades.
- `attendance_pct` and `gpa` may also correlate positively, as regular attendance likely supports academic performance.
- **Caveat**: correlation is not causation. A third variable (e.g., motivation or socioeconomic background) could drive both study hours and GPA simultaneously.

## 4. Hypothesis Test Results

### Hypothesis 1 — Internship students have higher GPA

- **Test**: Independent samples one-tailed t-test
- **t-statistic**: 13.5644
- **p-value**: 0.0000
- **Cohen's d**: 0.7061 (large effect)
- **Interpretation**: Statistically significant (p < 0.05). Internship students have a meaningfully higher GPA.

### Hypothesis 2 — Scholarship status is associated with department

- **Test**: Chi-square test of independence
- **Chi-square statistic**: 17.1358
- **p-value**: 0.3769
- **Degrees of freedom**: 16
- **Interpretation**: Not statistically significant (p >= 0.05). No strong evidence of association between scholarship and department.

## 5. Actionable Recommendations

1. **Promote internship programmes university-wide.** The t-test suggests internship students may have higher GPAs. Expanding access — especially in departments with low internship rates — could lift overall academic performance.

2. **Investigate and reduce scholarship data gaps.** Nearly 20% of scholarship records are missing. The university should audit its data collection process to ensure scholarship status is recorded for all students, enabling fairer financial aid analysis.

3. **Target support at low-attendance students.** The left-skewed attendance distribution reveals a distinct group attending below 70%. Early-warning interventions (e.g., advisor outreach) for these students could prevent GPA decline. See `output/dist_attendance.png`.
### Hypothesis 3 — GPA differs across departments (ANOVA)

- **Test**: One-way ANOVA
- **F-statistic**: 0.6671
- **p-value**: 0.6148
- **Interpretation**: Not statistically significant (p >= 0.05). GPA is similar across all departments.

