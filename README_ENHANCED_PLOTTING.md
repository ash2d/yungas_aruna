# Enhanced Ordinal Logistic Model Plotting

This directory contains enhanced plotting functionality for comparing full and restricted ordinal logistic models as used in the Granger causality analysis.

## Files

### 1. `enhanced_ordinal_model_plotting.py`
A standalone Python script that:
- Generates sample data matching the structure from Granger.ipynb
- Fits both full and restricted ordinal logistic models
- Creates comprehensive plots comparing the models
- Includes all requested features: unthresholded/thresholded data, MAE, and RMSE

### 2. `enhanced_granger_plotting.ipynb`
A Jupyter notebook that:
- Integrates with the existing Granger.ipynb analysis
- Provides the same enhanced plotting functionality
- Can be used with your actual data
- Contains detailed documentation and usage examples

### 3. `ordinal_model_predictions.png`
Example output plot showing the comprehensive model comparison.

## Key Features

### Full vs Restricted Model Comparison
- **Full Model**: Includes own lags + cross-species lags + exogenous variables (weather, diurnal)
- **Restricted Model**: Includes only own lags + exogenous variables (no cross-species interactions)

### Plot Layout (4×2 subplots)
1. **Row 1**: Unthresholded Full Model predictions vs actual data
2. **Row 2**: Unthresholded Restricted Model predictions vs actual data  
3. **Row 3**: Thresholded Full Model predictions vs actual data (binary 0/1)
4. **Row 4**: Thresholded Restricted Model predictions vs actual data (binary 0/1)

**Columns**: Oreo→Gastro direction | Gastro→Oreo direction

### Metrics Provided
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error) 
- **AIC** (Akaike Information Criterion)
- **LR Test** (Likelihood Ratio test for cross-species effects)
- **Model Selection** (which model is better by AIC)

## Usage

### Option 1: Standalone Script
```bash
# Run with sample data
python enhanced_ordinal_model_plotting.py
```

### Option 2: With Your Data
```python
# In Python/Jupyter
from enhanced_ordinal_model_plotting import fit_full_and_restricted_models, plot_model_predictions

# Assuming you have your DataFrame 'df' from Granger.ipynb
results = fit_full_and_restricted_models(
    data=df,
    gastro_col="Gastrotheca chysosticta",
    oreo_col="Oreobates berdemenos",
    temp_col="Temp", 
    rh_col="RH%",
    p=6,      # Use selected p from your granger analysis
    s_exog=6, # Weather lag span
    distr="logit"
)

plot_model_predictions(results, save_plot=True)
```

### Option 3: Jupyter Notebook
Open `enhanced_granger_plotting.ipynb` and follow the examples there.

## Interpretation

### Understanding the Plots
- **Black lines**: Actual observed data
- **Colored solid lines**: Full model predictions
- **Colored dashed lines**: Restricted model predictions

### Model Performance
- **Lower MAE/RMSE**: Better predictive accuracy
- **Lower AIC**: Better model considering complexity trade-off
- **LR Test p < 0.05**: Cross-species effects are statistically significant

### Thresholded vs Unthresholded
- **Unthresholded**: Shows how well models predict continuous expected values
- **Thresholded**: Shows binary classification performance (calls vs no calls)

## Example Output

The script generates output like:
```
MODEL COMPARISON SUMMARY
================================================================================

Oreobates berdemenos → Gastrotheca chysosticta:
  Full Model    - MAE: 0.5295, RMSE: 0.6794, AIC: 2409.76
  Restricted Model - MAE: 0.5348, RMSE: 0.6809, AIC: 2408.49
  LR Test: χ²=10.734, p=0.0970, df=6
  Better Model by AIC: Restricted

Gastrotheca chysosticta → Oreobates berdemenos:
  Full Model    - MAE: 0.3950, RMSE: 0.5679, AIC: 1886.95
  Restricted Model - MAE: 0.3992, RMSE: 0.5705, AIC: 1888.63
  LR Test: χ²=13.677, p=0.0335, df=6
  Better Model by AIC: Full
```

This shows that:
- For Oreo→Gastro: Restricted model is better (no significant cross-species effect)
- For Gastro→Oreo: Full model is better (significant cross-species effect, p=0.0335)

## Requirements

The code requires the same dependencies as Granger.ipynb:
- numpy
- pandas  
- matplotlib
- scipy
- statsmodels

## Integration with Existing Analysis

This enhanced plotting functionality is designed to extend your existing Granger causality analysis by providing:
1. Direct comparison between full and restricted models
2. Visual assessment of model fit quality
3. Statistical testing of cross-species interactions
4. Performance metrics for model selection

The plots help answer questions like:
- Do cross-species interactions improve prediction accuracy?
- Which species shows stronger evidence of Granger causality?
- How well do the models capture temporal patterns in the data?