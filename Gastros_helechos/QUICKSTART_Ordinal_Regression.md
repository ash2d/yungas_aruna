# Ordinal Regression Analysis - Quick Start Guide

## What This Does

Analyzes amphibian call intensities (0-3 scale) using **ordinal logistic regression** with:
- ✅ Nonlinear environmental effects (splines)
- ✅ Cyclic temporal patterns (Fourier terms)  
- ✅ Serial dependence (own lags)
- ✅ Cross-species influence (other species' lags)

## Quick Start

```bash
cd Gastros_helechos
python ordinal_regression_analysis.py
```

## Output

1. **Console**: Model summaries, coefficients, validity diagnostics
2. **Plot**: `plots/ordinal_regression_comprehensive.png` (10-panel visualization)

## Understanding the Results

### Model Performance (from sample data)
- **Oreobates**: AIC=5150, MAE=0.47
- **Gastrotheca**: AIC=6024, MAE=0.59
- **Convergence**: ✓ Both models
- **Autocorrelation**: ✓ Low (ACF ≈ 0)
- **Sample Size**: ✓ Adequate (116:1 obs/param ratio)

### Key Findings
1. **Environmental Effects**: Strong nonlinear relationships
   - Oreobates: Higher RH → more calls (OR=2-4 for RH splines)
   - Gastrotheca: Higher temp → more calls (OR=2-4 for temp splines)

2. **Temporal Patterns**: Clear diurnal and seasonal cycles
   - See bottom panels of plot

3. **Cross-Species**: No significant interaction in sample data
   - p = 0.34 for Gastro→Oreo
   - p = 0.43 for Oreo→Gastro

### Important Caveat ⚠️
**These models show ASSOCIATION, not CAUSATION!**
- Temporal lags suggest potential causality but don't prove it
- Cross-species effects could be confounded by unmeasured variables
- See full README for detailed caveats

## The Plot Explained

**Row 1 (top)**: Oreobates predictions vs actual data + residuals  
**Row 2**: Gastrotheca predictions vs actual data + residuals  
**Row 3**: Probability heatmaps (Temp × RH) at 22:00, mid-year  
**Row 4**: Diurnal patterns, seasonal patterns, performance metrics

## Customization

### Change spline flexibility
```python
df, spline_info = create_spline_features(df, n_knots=7)  # More knots = more flexible
```

### Add more lags
```python
df = create_lag_features(df, max_lag=3)  # Use lags 1, 2, 3
```

### Use probit instead of logit
```python
result = fit_ordinal_model(df, 'oreo_calls', predictors, distr='probit')
```

## Files

- `ordinal_regression_analysis.py` - Main script (1000+ lines)
- `README_Ordinal_Regression.md` - Full documentation
- `QUICKSTART.md` - This file
- `plots/ordinal_regression_comprehensive.png` - Output visualization

## Dependencies

```
numpy, pandas, matplotlib, scipy, statsmodels, patsy, scikit-learn, seaborn
```

Install: `pip install numpy pandas matplotlib scipy statsmodels patsy scikit-learn seaborn`

## Runtime

~30-60 seconds for 2900 observations

## Next Steps

1. Read the full README for detailed interpretation
2. Examine the comprehensive plot
3. Review coefficient tables in console output
4. Check validity assessment section
5. Consider cross-validation for robustness

---

For detailed documentation, see `README_Ordinal_Regression.md`
