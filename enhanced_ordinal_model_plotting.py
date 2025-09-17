#!/usr/bin/env python3
"""
Enhanced ordinal logistic model plotting script

This script creates both full and restricted ordinal logistic models
and plots their predictions against actual data, including both
unthresholded and thresholded data with MAE and RMSE metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import chi2
from math import sqrt


def create_sample_data(n_hours: int = 2000) -> pd.DataFrame:
    """Create sample data matching the structure from Granger.ipynb"""
    
    # Create hourly datetime index
    start_date = '2018-09-02 00:00:00'
    dates = pd.date_range(start=start_date, periods=n_hours, freq='h')
    
    # Create base DataFrame
    df = pd.DataFrame(index=dates)
    df.index.name = 'DateTime'
    
    # Add basic time columns
    df['Date'] = df.index.date
    df['Month'] = df.index.month
    df['Day'] = df.index.dayofyear
    df['hour'] = df.index.hour
    
    # Create realistic temperature and humidity patterns
    # Temperature varies with hour of day and some noise
    hour_temp_effect = 2 * np.sin(2 * np.pi * df['hour'] / 24 - np.pi/2)  # Peak at 2pm
    daily_temp_variation = np.random.normal(0, 1, len(df))
    df['Temp'] = 5 + hour_temp_effect + daily_temp_variation
    
    # Humidity inversely related to temperature with noise
    df['RH%'] = np.clip(95 - 2 * hour_temp_effect + np.random.normal(0, 5, len(df)), 60, 100)
    
    # Create call counts that depend on weather and time
    # Higher probability of calls during certain hours and weather conditions
    evening_hours = ((df['hour'] >= 18) | (df['hour'] <= 6)).astype(int)
    weather_effect = (df['Temp'] > 4) & (df['RH%'] > 85)
    
    # Base probabilities influenced by weather and time
    gastro_prob = 0.15 + 0.1 * evening_hours + 0.05 * weather_effect.astype(int)
    oreo_prob = 0.12 + 0.08 * evening_hours + 0.03 * weather_effect.astype(int)
    
    # Add some cross-species correlation (one species might influence another)
    # Generate calls with some temporal correlation
    gastro_calls = []
    oreo_calls = []
    
    for i in range(len(df)):
        # Previous hour influence
        prev_gastro = gastro_calls[i-1] if i > 0 else 0
        prev_oreo = oreo_calls[i-1] if i > 0 else 0
        
        # Current probabilities adjusted by previous calls
        current_gastro_prob = gastro_prob.iloc[i] + 0.1 * (prev_oreo > 0)
        current_oreo_prob = oreo_prob.iloc[i] + 0.08 * (prev_gastro > 0)
        
        # Generate calls (0-3 scale as in original data)
        gastro_calls.append(np.random.choice([0, 1, 2, 3], 
                                           p=[1-current_gastro_prob, 
                                              current_gastro_prob*0.7, 
                                              current_gastro_prob*0.25, 
                                              current_gastro_prob*0.05]))
        
        oreo_calls.append(np.random.choice([0, 1, 2, 3], 
                                         p=[1-current_oreo_prob, 
                                            current_oreo_prob*0.75, 
                                            current_oreo_prob*0.2, 
                                            current_oreo_prob*0.05]))
    
    df['Gastrotheca chysosticta'] = gastro_calls
    df['Oreobates berdemenos'] = oreo_calls
    
    return df


# Copy the utility functions from Granger.ipynb
def add_time_features(dfh: pd.DataFrame) -> pd.DataFrame:
    """Add diurnal sin/cos based on the DateTimeIndex (hour + minute)."""
    h = dfh.index.hour + dfh.index.minute / 60.0
    dfh = dfh.copy()
    dfh["hour_sin"] = np.sin(2 * np.pi * h / 24)
    dfh["hour_cos"] = np.cos(2 * np.pi * h / 24)
    
    # Add day of year features
    d = dfh.index.dayofyear
    dfh["day_sin"] = np.sin(2 * np.pi * d / 365)
    dfh["day_cos"] = np.cos(2 * np.pi * d / 365)
    
    return dfh


def zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Z-score specified columns."""
    df = df.copy()
    for c in cols:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()
    return df


def make_lags(df: pd.DataFrame, cols: List[str], L: int) -> pd.DataFrame:
    """Create lagged columns col_L1..col_LL for each col."""
    out = {}
    for c in cols:
        for l in range(1, L + 1):
            out[f"{c}_L{l}"] = df[c].shift(l)
    return pd.DataFrame(out, index=df.index)


def aic_of_fit(res) -> float:
    """Compute AIC for OrderedModel fit result."""
    k = len(res.params)
    return -2 * res.llf + 2 * k


def fit_ordered(endog: pd.Series, exog: pd.DataFrame, distr: str = "logit"):
    """
    Fit ordered logit/probit without an explicit constant (thresholds play that role).
    endog must be integer categories (0..K-1).
    """
    # Drop any rows with missing exog (endog should already be aligned)
    mask = endog.notna()
    mask &= exog.notna().all(axis=1)
    y = endog[mask].astype(int)
    X = exog.loc[mask]

    # Quick rank check to catch collinearity early (optional)
    rank = np.linalg.matrix_rank(X.values)
    if rank < X.shape[1]:
        raise ValueError(f"Design matrix not full rank (rank={rank} < {X.shape[1]}). "
                         f"Reduce lags / remove redundant columns (do not lag sin/cos).")

    model = OrderedModel(y, X, distr=distr)
    # Increase maxiter if needed; method can be 'bfgs'/'lbfgs'/'newton'
    res = model.fit(method="lbfgs", maxiter=2000, disp=False)
    return res, y.index


def lr_test(res_full, res_rest, df_restr: int) -> Tuple[float, float]:
    """LR stat and chi-square p-value."""
    LR = 2 * (res_full.llf - res_rest.llf)
    p = 1 - chi2.cdf(LR, df_restr)
    return LR, p


def fit_full_and_restricted_models(data: pd.DataFrame, 
                                   gastro_col: str, 
                                   oreo_col: str, 
                                   temp_col: str, 
                                   rh_col: str,
                                   p: int = 6,
                                   s_exog: int = 6,
                                   distr: str = "logit") -> Dict:
    """
    Fit both full and restricted ordinal logistic models for both species.
    
    Returns:
        Dictionary containing full and restricted model results for both directions
    """
    
    # Prepare data
    req = [gastro_col, oreo_col, temp_col, rh_col]
    data = data.sort_index()[req].copy()
    
    # Ensure integer categories
    for c in (gastro_col, oreo_col):
        data[c] = data[c].astype(int)
    
    # Add time features and standardize weather
    data = add_time_features(data)
    data = zscore(data, [temp_col, rh_col])
    
    # Build EXOG matrix (diurnal + weather lags)
    EXOG_parts = [data[["hour_sin", "hour_cos", "day_sin", "day_cos"]]]
    for base in [temp_col + "_z", rh_col + "_z"]:
        cols_ex = {f"{base}_L0": data[base]}
        for L in range(1, s_exog + 1):
            cols_ex[f"{base}_L{L}"] = data[base].shift(L)
        EXOG_parts.append(pd.DataFrame(cols_ex, index=data.index))
    EXOG = pd.concat(EXOG_parts, axis=1)
    
    def fit_direction_models(Y_name: str, X_name: str):
        """Fit both full and restricted models for a given direction."""
        
        # Build lag matrices
        own_lags = make_lags(data, [Y_name], p)
        cross_lags = make_lags(data, [X_name], p)
        
        # Full design = own lags + cross lags + exogenous
        X_full = pd.concat([own_lags, cross_lags, EXOG], axis=1)
        
        # Restricted design = own lags + exogenous (no cross lags)
        X_rest = pd.concat([own_lags, EXOG], axis=1)
        
        endog = data[Y_name]
        
        # Fit both models
        res_full, idx_full = fit_ordered(endog, X_full, distr=distr)
        res_rest, _ = fit_ordered(endog.loc[idx_full], X_rest.loc[idx_full], distr=distr)
        
        # Generate predictions
        probs_full = res_full.predict(exog=X_full.loc[idx_full])
        probs_rest = res_rest.predict(exog=X_rest.loc[idx_full])
        
        # Get category labels and compute expected values
        cats = np.sort(endog.loc[idx_full].unique())
        exp_vals_full = (probs_full * cats).sum(axis=1)
        exp_vals_rest = (probs_rest * cats).sum(axis=1)
        
        # Compute metrics
        actual_vals = endog.loc[idx_full]
        
        # Full model metrics
        resid_full = actual_vals - exp_vals_full
        mae_full = float(np.abs(resid_full).mean())
        rmse_full = float(sqrt((resid_full**2).mean()))
        
        # Restricted model metrics
        resid_rest = actual_vals - exp_vals_rest
        mae_rest = float(np.abs(resid_rest).mean())
        rmse_rest = float(sqrt((resid_rest**2).mean()))
        
        # LR test
        LR, pval = lr_test(res_full, res_rest, p)  # p cross-lags removed
        
        return {
            'direction': f"{X_name} → {Y_name}",
            'full_model': {
                'predictions': exp_vals_full,
                'probabilities': probs_full,
                'MAE': mae_full,
                'RMSE': rmse_full,
                'AIC': aic_of_fit(res_full),
                'result': res_full
            },
            'restricted_model': {
                'predictions': exp_vals_rest,
                'probabilities': probs_rest,
                'MAE': mae_rest,
                'RMSE': rmse_rest,
                'AIC': aic_of_fit(res_rest),
                'result': res_rest
            },
            'actual_values': actual_vals,
            'index': idx_full,
            'LR_test': {'statistic': float(LR), 'p_value': float(pval), 'df': p}
        }
    
    # Fit both directions
    oreo_to_gastro = fit_direction_models(gastro_col, oreo_col)
    gastro_to_oreo = fit_direction_models(oreo_col, gastro_col)
    
    return {
        'Oreo_to_Gastro': oreo_to_gastro,
        'Gastro_to_Oreo': gastro_to_oreo,
        'settings': {
            'p': p,
            's_exog': s_exog,
            'distr': distr
        }
    }


def create_threshold_data(data: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Create thresholded version of continuous data."""
    return (data >= threshold).astype(int)


def plot_model_predictions(results: Dict, 
                          gastro_col: str = "Gastrotheca chysosticta",
                          oreo_col: str = "Oreobates berdemenos",
                          save_plot: bool = True,
                          plot_filename: str = "ordinal_model_predictions.png"):
    """
    Plot full and restricted model predictions against actual data.
    Includes both unthresholded and thresholded comparisons.
    """
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Ordinal Logistic Model Predictions: Full vs Restricted Models', fontsize=16)
    
    directions = [
        ('Oreo_to_Gastro', gastro_col, 'tab:green'),
        ('Gastro_to_Oreo', oreo_col, 'tab:orange')
    ]
    
    for col_idx, (direction_key, species_col, color) in enumerate(directions):
        direction_results = results[direction_key]
        
        actual = direction_results['actual_values']
        pred_full = direction_results['full_model']['predictions']
        pred_rest = direction_results['restricted_model']['predictions']
        idx = direction_results['index']
        
        # Create thresholded versions (threshold at 0.5)
        actual_thresh = create_threshold_data(actual, 0.5)
        pred_full_thresh = create_threshold_data(pred_full, 0.5)
        pred_rest_thresh = create_threshold_data(pred_rest, 0.5)
        
        # Plot 1: Unthresholded actual vs full model
        ax1 = axes[0, col_idx]
        ax1.plot(idx, actual, 'k-', alpha=0.7, linewidth=0.8, label='Actual')
        ax1.plot(idx, pred_full, color=color, linewidth=1.0, label='Full Model')
        ax1.set_title(f'{direction_results["direction"]} - Full Model (Unthresholded)')
        ax1.set_ylabel('Call Count')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Add metrics text
        full_mae = direction_results['full_model']['MAE']
        full_rmse = direction_results['full_model']['RMSE']
        ax1.text(0.02, 0.98, f'MAE: {full_mae:.3f}\nRMSE: {full_rmse:.3f}', 
                transform=ax1.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Unthresholded actual vs restricted model
        ax2 = axes[1, col_idx]
        ax2.plot(idx, actual, 'k-', alpha=0.7, linewidth=0.8, label='Actual')
        ax2.plot(idx, pred_rest, color=color, linewidth=1.0, alpha=0.7, 
                linestyle='--', label='Restricted Model')
        ax2.set_title(f'{direction_results["direction"]} - Restricted Model (Unthresholded)')
        ax2.set_ylabel('Call Count')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add metrics text
        rest_mae = direction_results['restricted_model']['MAE']
        rest_rmse = direction_results['restricted_model']['RMSE']
        ax2.text(0.02, 0.98, f'MAE: {rest_mae:.3f}\nRMSE: {rest_rmse:.3f}', 
                transform=ax2.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Thresholded actual vs full model
        ax3 = axes[2, col_idx]
        ax3.plot(idx, actual_thresh, 'k-', alpha=0.7, linewidth=0.8, label='Actual (Thresholded)')
        ax3.plot(idx, pred_full_thresh, color=color, linewidth=1.0, label='Full Model (Thresholded)')
        ax3.set_title(f'{direction_results["direction"]} - Full Model (Thresholded at 0.5)')
        ax3.set_ylabel('Binary Calls (0/1)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Compute binary metrics
        full_mae_thresh = float(np.abs(actual_thresh - pred_full_thresh).mean())
        full_rmse_thresh = float(sqrt(((actual_thresh - pred_full_thresh)**2).mean()))
        ax3.text(0.02, 0.98, f'MAE: {full_mae_thresh:.3f}\nRMSE: {full_rmse_thresh:.3f}', 
                transform=ax3.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Thresholded actual vs restricted model
        ax4 = axes[3, col_idx]
        ax4.plot(idx, actual_thresh, 'k-', alpha=0.7, linewidth=0.8, label='Actual (Thresholded)')
        ax4.plot(idx, pred_rest_thresh, color=color, linewidth=1.0, alpha=0.7, 
                linestyle='--', label='Restricted Model (Thresholded)')
        ax4.set_title(f'{direction_results["direction"]} - Restricted Model (Thresholded at 0.5)')
        ax4.set_ylabel('Binary Calls (0/1)')
        ax4.set_xlabel('Time')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # Compute binary metrics
        rest_mae_thresh = float(np.abs(actual_thresh - pred_rest_thresh).mean())
        rest_rmse_thresh = float(sqrt(((actual_thresh - pred_rest_thresh)**2).mean()))
        ax4.text(0.02, 0.98, f'MAE: {rest_mae_thresh:.3f}\nRMSE: {rest_rmse_thresh:.3f}', 
                transform=ax4.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for direction_key in ['Oreo_to_Gastro', 'Gastro_to_Oreo']:
        result = results[direction_key]
        print(f"\n{result['direction']}:")
        print(f"  Full Model    - MAE: {result['full_model']['MAE']:.4f}, RMSE: {result['full_model']['RMSE']:.4f}, AIC: {result['full_model']['AIC']:.2f}")
        print(f"  Restricted Model - MAE: {result['restricted_model']['MAE']:.4f}, RMSE: {result['restricted_model']['RMSE']:.4f}, AIC: {result['restricted_model']['AIC']:.2f}")
        print(f"  LR Test: χ²={result['LR_test']['statistic']:.3f}, p={result['LR_test']['p_value']:.4f}, df={result['LR_test']['df']}")
        
        # Determine which model is better
        full_aic = result['full_model']['AIC']
        rest_aic = result['restricted_model']['AIC']
        better_model = "Full" if full_aic < rest_aic else "Restricted"
        print(f"  Better Model by AIC: {better_model}")


def main():
    """Main function to run the enhanced ordinal model analysis."""
    
    print("Creating sample data...")
    df = create_sample_data(n_hours=1500)
    
    print("Sample data structure:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    print("\nFitting full and restricted ordinal logistic models...")
    results = fit_full_and_restricted_models(
        data=df,
        gastro_col="Gastrotheca chysosticta",
        oreo_col="Oreobates berdemenos", 
        temp_col="Temp",
        rh_col="RH%",
        p=6,  # 6 hours of lags
        s_exog=6,  # 6 hours of weather lags
        distr="logit"
    )
    
    print("\nCreating enhanced plots...")
    plot_model_predictions(
        results=results,
        save_plot=True,
        plot_filename="/home/runner/work/yungas_aruna/yungas_aruna/ordinal_model_predictions.png"
    )


if __name__ == "__main__":
    main()