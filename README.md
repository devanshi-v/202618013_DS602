# Medical Insurance Cost Analysis Dashboard

An end-to-end Data Science project combining **exploratory data analysis, statistical hypothesis testing, multiple linear regression, model diagnostics, and an interactive Streamlit dashboard** using the Medical Insurance Costs dataset.

## Project Overview

This project analyzes medical insurance charges using demographic, health, and policy-related variables. It identifies factors associated with charges, tests differences between groups, fits an OLS regression model, evaluates model assumptions, and presents the results through an interactive dashboard.

## Objectives

- Perform descriptive and exploratory statistical analysis.
- Examine relationships between numerical and categorical variables.
- Apply parametric and non-parametric hypothesis tests.
- Fit and interpret a Multiple Linear Regression model using `statsmodels.api.OLS`.
- Evaluate residuals and multicollinearity.
- Build an interactive Streamlit dashboard with live predictions and a 95% prediction interval.

## Dataset

The project uses the **Medical Insurance Costs** dataset.

### Dataset Summary

| Item | Details |
|---|---|
| Original observations | 1,338 |
| Variables | 7 |
| Duplicate rows | 1 |
| Observations after cleaning | 1,337 |
| Missing values | None |
| Target variable | `charges` |

### Features

| Feature | Description | Type |
|---|---|---|
| age | Age of the policyholder | Numerical |
| sex | Sex of the policyholder | Categorical |
| bmi | Body Mass Index | Numerical |
| children | Number of children/dependents | Numerical |
| smoker | Smoking status | Categorical |
| region | Residential region | Categorical |
| charges | Medical insurance charges | Numerical / Target |

## Exploratory Data Analysis

The analysis includes descriptive statistics, histograms with KDE, box plots, scatter plots, categorical analysis, and a correlation heatmap.

### Key Descriptive Statistics

| Variable | Mean | Median | Std. Dev. | IQR | Skewness | Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| Age | 39.22 | 39.00 | 14.04 | 24.00 | 0.055 | -1.244 |
| BMI | 30.66 | 30.40 | 6.10 | 8.41 | 0.284 | -0.053 |
| Children | 1.10 | 1.00 | 1.21 | 2.00 | 0.937 | 0.201 |
| Charges | 13,279.12 | 9,386.16 | 12,110.36 | 11,911.37 | 1.515 | 1.604 |

Medical charges show noticeable positive skewness, with skewness of approximately **1.515**.

## Hypothesis Testing

All hypothesis tests use a significance level of **α = 0.05**.

### 1. Smokers vs Non-Smokers

**H₀:** Mean medical charges are the same for smokers and non-smokers.  
**H₁:** Mean medical charges are different for smokers and non-smokers.

Shapiro-Wilk and Levene's tests were used to check normality and variance assumptions. Since the data were not normally distributed, the **Mann-Whitney U test** was selected.

- **U statistic:** 283,859.0
- **p-value:** 5.747 × 10⁻¹³⁰
- **Decision:** Reject H₀

There is a statistically significant difference in medical charges between smokers and non-smokers.

### 2. Charges Across Regions

A **One-Way ANOVA** was used to compare average medical charges across the four regions.

**H₀:** Mean medical charges are equal across all regions.  
**H₁:** At least one region has a different mean medical charge.

- **F-statistic:** 2.9261
- **p-value:** 0.03276
- **Decision:** Reject H₀

Average medical charges differ significantly across regions at the 5% significance level.

## Multiple Linear Regression

A Multiple Linear Regression model was fitted using `statsmodels.api.OLS` after one-hot encoding categorical variables.

**Model:**

text
charges ~ age + bmi + children + sex + smoker + region

### Model Performance

| Metric | Value |
|---|---:|
| R² | 0.751 |
| Adjusted R² | 0.749 |
| F-statistic | ≈ 500.0 |
| Overall model p-value | < 0.001 |

The model explains approximately **75.1% of the variation in medical insurance charges**.

### Important Coefficients

| Predictor | Coefficient | Interpretation |
|---|---:|---|
| Age | 256.76 | Charges increase as age increases |
| BMI | 339.25 | Higher BMI is associated with higher charges |
| Children | 474.82 | Positive association with charges |
| Smoker | 23,850 | Strong positive association with charges |
| Sex | -129.48 | Not statistically significant |
| Region | — | Southeast and Southwest are significant relative to the reference region |

## Regression Diagnostics

- **Residuals vs Fitted:** Used to inspect linearity and changing variance.
- **Q-Q Plot:** Used to assess residual normality.
- **Jarque-Bera:** Statistic = **716.55**, p-value = **2.527 × 10⁻¹⁵⁶**, indicating significant deviation from normality.
- **VIF:** Maximum VIF is approximately **1.65**, indicating no serious multicollinearity.

## Streamlit Dashboard

The interactive dashboard is organized into three tabs.

### Tab 1 — Data Exploration

- Dataset overview and summary statistics
- Interactive age, smoker, and region filters
- Distribution and box plots
- Scatter plots
- Correlation heatmap

### Tab 2 — Hypothesis Testing Lab

Users can select:

1. **Smokers vs Non-Smokers**
2. **Charges Across Regions (ANOVA)**

The dashboard performs the required tests automatically and displays the test statistic, p-value, visualization, and conclusion.

### Tab 3 — Live Prediction & Diagnostics

Users can enter age, sex, BMI, smoking status, number of children, and region to obtain:

- Predicted medical charges
- **95% prediction interval**
- R² and Adjusted R²
- Model significance
- Residuals vs Fitted plot
- Q-Q plot
- VIF table
- OLS regression summary

## Key Findings

1. Smoking status has a strong association with medical insurance charges.
2. Charges differ significantly between smokers and non-smokers.
3. Average charges differ significantly across regions.
4. The OLS model explains approximately **75.1%** of the variation in charges.
5. Age, BMI, children, and smoking status are significant predictors, while sex is not significant at the 5% level.
6. Multicollinearity is not a major concern.
7. Residuals show significant non-normality according to the Jarque-Bera test.
