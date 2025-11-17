# Ordinal Regression Analysis for Amphibian Call Intensities

## Overview

This directory contains a comprehensive end-to-end implementation of cumulative-link (ordinal) regression models for analyzing amphibian call intensities. The analysis models call behavior on a 0-3 scale for two species:
- **Oreobates berdemenos**
- **Gastrotheca chysosticta**

## Statistical Framework

### Model Specification

The analysis implements **ordinal logistic regression** (cumulative-link models) with the following key features:

1. **Nonlinearity in Environmental Variables**
   - Natural cubic splines for Temperature and Relative Humidity
   - 5 basis functions each (6 knots with 1 dropped to avoid constant sum)
   - Captures complex nonlinear relationships

2. **Cyclic Seasonality**
   - Fourier terms for hour-of-day (2 harmonics = 4 features)
   - Fourier terms for day-of-year (3 harmonics = 6 features)
   - Models diurnal and seasonal patterns without edge effects

3. **Serial Dependence**
   - Lag-1 of own species' calls
   - Accounts for temporal autocorrelation
   - Prevents confounding environmental effects with persistence

4. **Cross-Species Influence**
   - Lag-1 of other species' calls
   - Tests for behavioral association (not causation!)
   - Allows detection of potential inter-species interactions

### Mathematical Form

For species $i$ at time $t$:

$$P(Y_{i,t} \leq j) = \text{logit}^{-1}\left(\theta_j - \left[\beta_1 Y_{i,t-1} + \beta_2 Y_{j,t-1} + f_{\text{temp}}(T_t) + f_{\text{rh}}(RH_t) + f_{\text{time}}(t)\right]\right)$$

Where:
- $Y_{i,t}$ = call intensity (0-3) for species $i$ at time $t$
- $\theta_j$ = threshold parameters for ordinal categories
- $Y_{i,t-1}$ = own lag
- $Y_{j,t-1}$ = other species' lag
- $f_{\text{temp}}, f_{\text{rh}}$ = spline transformations
- $f_{\text{time}}$ = Fourier seasonal/diurnal terms

## File Structure

```
Gastros_helechos/
├── ordinal_regression_analysis.py  # Main analysis script
├── README_Ordinal_Regression.md    # This file
├── sample_helechos_data.csv        # Sample data
└── plots/
    └── ordinal_regression_comprehensive.png  # Output visualization
```

## Usage

### Running the Analysis

```bash
cd Gastros_helechos
python ordinal_regression_analysis.py
```

The script will:
1. Load and tidy the data (binning calls to 0-3 if needed)
2. Create cyclic (Fourier) features
3. Create spline features for environmental variables
4. Add lag features for serial and cross-species dependence
5. Fit ordinal models for both species
6. Generate comprehensive visualizations
7. Assess model validity and provide interpretations

### Expected Output

The script produces:
- **Console output**: Model summaries, coefficient interpretations, validity assessments
- **Visualization**: `plots/ordinal_regression_comprehensive.png` with 10 subplots showing:
  - Time series predictions vs actual data (both species)
  - Residual distributions with MAE metrics
  - Probability heatmaps over Temperature-Humidity space
  - Cross-species interaction effects
  - Diurnal and seasonal patterns
  - Model performance metrics

## Interpretation Guide

### 1. Coefficient Interpretation

**Odds Ratios**: For a predictor $X$:
- OR > 1: Positive effect (increases probability of higher call intensity)
- OR < 1: Negative effect (decreases probability of higher call intensity)
- OR = $e^{\beta}$ where $\beta$ is the coefficient

**Example**:
```
rh_spline_5: Coefficient = 1.485, Odds Ratio = 4.41, p < 0.001
```
This means: Higher values of this RH spline basis increase the odds of higher call intensity by a factor of 4.41.

### 2. Environmental Effects

Temperature and RH effects are **nonlinear** and captured through splines:
- Check the probability heatmaps to see optimal conditions
- Spline coefficients show piece-wise effects
- Look for clustering of significant spline terms

**Key Finding from Sample Data**:
- Oreobates: Higher RH strongly associated with increased calling (all RH spline terms positive and significant)
- Gastrotheca: Temperature has positive effect (temp splines mostly positive), RH has negative effect

### 3. Temporal Patterns

**Diurnal Patterns** (hour-of-day Fourier terms):
- Check diurnal pattern plot (bottom left)
- Both species show peak calling at specific hours
- Captured by sin/cos harmonics

**Seasonal Patterns** (day-of-year Fourier terms):
- Check seasonal pattern plot (bottom middle)
- Shows year-round variation in calling activity

### 4. Serial Dependence vs Cross-Species Effects

**Serial Dependence** (own lag):
- Tests if previous hour's calls predict current calls
- In sample data: Not strongly significant (p > 0.05)
- Suggests calling is more environmentally driven than persistent

**Cross-Species Influence**:
- Tests if other species' calls predict current calls
- In sample data: Not significant (p = 0.34 for Gastro→Oreo, p = 0.43 for Oreo→Gastro)
- **Important**: Non-significance doesn't mean no interaction, just that it's not detected above environmental correlations

### 5. Model Performance Metrics

**AIC/BIC**: Lower is better
- Oreobates: AIC = 5150, BIC = 5300
- Gastrotheca: AIC = 6024, BIC = 6173

**MAE (Mean Absolute Error)**:
- Oreobates: ~0.47 (on 0-3 scale)
- Gastrotheca: ~0.59 (on 0-3 scale)
- Interpretation: Predictions are typically off by < 0.6 intensity units

**Residual Autocorrelation**:
- ACF(1) ≈ 0 for both species (✓)
- Confirms lags adequately capture temporal dependence

## Validity Assessment

### 1. Model Convergence ✓
Both models converged successfully.

### 2. Proportional Odds Assumption ⚠
The model assumes effects are consistent across all threshold levels. If this is violated, consider:
- Partial proportional odds models
- Multinomial logistic regression

### 3. Multicollinearity ✓
Condition number ≈ 44 (moderate, acceptable)
- Splines introduce some correlation but within acceptable range

### 4. Sample Size ✓
~116 observations per parameter
- Well above the 10-20:1 rule of thumb

### 5. Autocorrelation ✓
Residual ACF(1) ≈ 0
- Lag-1 predictors successfully capture temporal dependence

### 6. Out-of-Sample Validation ⚠
Not performed in this script
- **Recommendation**: Implement time-series cross-validation
- Hold out later time periods for testing

## Causal Interpretation - IMPORTANT CAVEATS ⚠

### What the Models Show
- **ASSOCIATION**, not causation
- Cross-species lags show correlation after accounting for environment
- Temporal precedence (lag) is suggestive but not proof

### What Could Explain Cross-Species Associations
1. **Shared environmental drivers**: Unmeasured variables affecting both species
2. **Common behavioral responses**: Both species respond to same cues
3. **True behavioral interaction**: One species influences the other
4. **Spatial correlation**: Both species use same microhabitats

### Requirements for Causal Claims
- **Experimental manipulation**: Playback experiments, removal studies
- **Mechanistic understanding**: Behavioral observations
- **Ruling out confounders**: More comprehensive environmental measurements

## Extending the Analysis

### 1. Adding More Predictors
```python
# In main(), before fitting models:
additional_predictors = ['rainfall', 'moon_phase', 'wind_speed']
oreo_predictors += additional_predictors
gastro_predictors += additional_predictors
```

### 2. Increasing Lag Order
```python
# Change max_lag in create_lag_features()
df = create_lag_features(df, max_lag=3)  # Use lags 1, 2, 3
```

### 3. More Spline Knots
```python
# In main():
df, spline_info = create_spline_features(df, n_knots=7)  # More flexible curves
```

### 4. Different Link Functions
```python
# Try probit instead of logit:
oreo_result, oreo_model, oreo_mask = fit_ordinal_model(
    df, 'oreo_calls', oreo_predictors, distr='probit'
)
```

## Computational Requirements

- **Python packages**: numpy, pandas, matplotlib, scipy, statsmodels, patsy, scikit-learn, seaborn
- **Memory**: ~200 MB for 2900 observations
- **Runtime**: ~30-60 seconds for full analysis
- **Output size**: ~1.3 MB plot file

## Data Requirements

The input CSV should have:
- **DateTime information**: Either 'DateTime' column or 'Date' + 'hour' columns
- **Species calls**: Columns with species names containing call counts (0-7 range, will be binned to 0-3)
- **Temperature**: Column named 'Temp' or specified in parameters
- **Relative Humidity**: Column named 'RH%' or specified in parameters

Example structure:
```csv
Date,Month,Day,hour,Temp,RH%,Oreobates berdemenos,Gastrotheca chysosticta
2018-01-01,1,1,6,17.14,80.31,0,2
2018-01-01,1,1,17,20.20,96.27,2,0
...
```

## References

### Statistical Methods
- **Ordinal Regression**: Agresti, A. (2010). Analysis of Ordinal Categorical Data. Wiley.
- **Splines**: Wood, S.N. (2017). Generalized Additive Models: An Introduction with R. CRC Press.
- **Time Series**: Shumway, R.H. & Stoffer, D.S. (2017). Time Series Analysis and Its Applications. Springer.

### Ecological Applications
- **Granger Causality**: See Granger_causality/ directory for related analyses
- **Behavioral Ecology**: Call intensity as a proxy for activity and reproductive behavior

## Troubleshooting

### Common Issues

**1. "ValueError: There should not be a constant in the model"**
- Solution: Script automatically removes constant columns (spline sum issue)
- Already handled in the code

**2. "Singular matrix" or convergence issues**
- Reduce number of spline knots
- Check for perfect collinearity in predictors
- Ensure sufficient data variation

**3. Poor model fit (high MAE)**
- Add more environmental predictors
- Increase spline flexibility
- Check for outliers or data quality issues

**4. Non-significant effects**
- May indicate true absence of effect
- Or insufficient power / model misspecification
- Consider longer time series or more frequent sampling

## Contact & Contributions

This analysis was developed for the Yungas-Aruna project studying amphibian behavior in tropical montane forests.

For questions or contributions, please open an issue in the repository.

---

**Last Updated**: November 2025
**Version**: 1.0
**License**: See repository LICENSE file
