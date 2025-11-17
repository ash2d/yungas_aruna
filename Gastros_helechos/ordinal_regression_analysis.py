#!/usr/bin/env python3
"""
End-to-End Ordinal Regression Modeling for Species Call Intensities

This script implements cumulative-link (ordinal) regression for modeling 
amphibian call intensities (0-3 scale) with:
- Nonlinearity in Temperature & Relative Humidity (splines)
- Cyclic seasonality (Fourier terms for hour-of-day and day-of-year)
- Serial dependence (lag of previous hour's call for same species)
- Cross-species influence (other species' lag(1) as predictor)

Models fitted:
1. Oreobates berdemenos call intensity
2. Gastrotheca chysosticta call intensity

Output:
- Model summaries and coefficients
- Predicted probabilities on Temp-Humidity grid
- Comprehensive plots showing:
  - Model predictions vs actual data
  - Partial dependence plots for key variables
  - Probability surfaces over Temp-RH space
- Interpretation of variables' impact on calls
- Assessment of inference validity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Tuple, List
from statsmodels.miscmodels.ordinal_model import OrderedModel
from patsy import dmatrix, bs, cr
from scipy.interpolate import BSpline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load data and prepare it for analysis.
    
    Args:
        filepath: Path to CSV data file
        
    Returns:
        DataFrame with proper datetime index and cleaned columns
    """
    print("=" * 80)
    print("STEP 1: LOADING AND TIDYING DATA")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\nLoaded {len(df)} observations")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create datetime index if not present
    if 'DateTime' not in df.columns:
        # Reconstruct datetime from Date and hour
        df['DateTime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['hour'], unit='h')
    else:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    df = df.set_index('DateTime').sort_index()
    
    # Standardize column names
    df = df.rename(columns={
        'Oreobates berdemenos': 'oreo_calls',
        'Gastrotheca chysosticta': 'gastro_calls',
        'Temp': 'temp',
        'RH%': 'rh'
    })
    
    # Ensure we have the required columns
    required_cols = ['temp', 'rh', 'oreo_calls', 'gastro_calls']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Handle call intensities - bin into 0-3 scale if needed
    print("\nCall intensity distribution:")
    for col in ['oreo_calls', 'gastro_calls']:
        max_val = df[col].max()
        print(f"  {col}: range 0-{max_val}")
        
        # If values exceed 3, bin them into 0-3 scale
        if max_val > 3:
            print(f"    Binning {col} into 0-3 scale...")
            # Use value-based binning: 0, 1-2, 3-4, 5+
            bins = [-0.5, 0.5, 2.5, 4.5, max_val + 0.5]
            df[col] = pd.cut(df[col], bins=bins, labels=[0, 1, 2, 3])
            df[col] = df[col].astype(int)
    
    print("\nFinal call intensity distribution:")
    for col in ['oreo_calls', 'gastro_calls']:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index())
    
    # Remove any rows with missing values
    df = df.dropna(subset=required_cols)
    print(f"\nFinal dataset: {len(df)} complete observations")
    
    return df


def create_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclic Fourier features for hour-of-day and day-of-year.
    
    Args:
        df: DataFrame with DateTimeIndex
        
    Returns:
        DataFrame with added Fourier features
    """
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING CYCLIC FEATURES (Fourier Terms)")
    print("=" * 80)
    
    df = df.copy()
    
    # Hour of day (0-23) - use 2 harmonics
    hour = df.index.hour + df.index.minute / 60.0
    for k in range(1, 3):  # 2 harmonics
        df[f'hour_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24)
        df[f'hour_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24)
    
    # Day of year (1-365) - use 3 harmonics
    doy = df.index.dayofyear
    for k in range(1, 4):  # 3 harmonics
        df[f'doy_sin_{k}'] = np.sin(2 * np.pi * k * doy / 365)
        df[f'doy_cos_{k}'] = np.cos(2 * np.pi * k * doy / 365)
    
    print("\nCreated Fourier features:")
    print("  Hour-of-day: 2 harmonics (4 features)")
    print("  Day-of-year: 3 harmonics (6 features)")
    print("  Total: 10 cyclic features")
    
    return df


def create_spline_features(df: pd.DataFrame, 
                          temp_col: str = 'temp',
                          rh_col: str = 'rh',
                          n_knots: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Create spline features for temperature and humidity.
    
    Args:
        df: Input DataFrame
        temp_col: Temperature column name
        rh_col: Relative humidity column name
        n_knots: Number of internal knots for splines
        
    Returns:
        DataFrame with spline features and dict of spline info
    """
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING SPLINE FEATURES (Nonlinearity)")
    print("=" * 80)
    
    df = df.copy()
    spline_info = {}
    
    # Standardize temperature and humidity for better numerical stability
    scaler_temp = StandardScaler()
    scaler_rh = StandardScaler()
    
    df['temp_std'] = scaler_temp.fit_transform(df[[temp_col]])
    df['rh_std'] = scaler_rh.fit_transform(df[[rh_col]])
    
    # Create natural cubic splines using patsy
    # Degrees of freedom = n_knots + 1 (for natural splines)
    df_splines = n_knots + 1
    
    # Temperature splines
    temp_formula = f"cr(temp_std, df={df_splines}) - 1"  # -1 removes intercept
    temp_splines = dmatrix(temp_formula, df, return_type='dataframe')
    temp_splines.columns = [f'temp_spline_{i}' for i in range(len(temp_splines.columns))]
    
    # RH splines
    rh_formula = f"cr(rh_std, df={df_splines}) - 1"
    rh_splines = dmatrix(rh_formula, df, return_type='dataframe')
    rh_splines.columns = [f'rh_spline_{i}' for i in range(len(rh_splines.columns))]
    
    # IMPORTANT: Natural splines sum to 1 (constant), which causes issues with OrderedModel
    # Drop the first basis function from each to remove the linear dependency
    temp_splines = temp_splines.iloc[:, 1:]  # Drop first column
    rh_splines = rh_splines.iloc[:, 1:]  # Drop first column
    
    # Add to dataframe
    df = pd.concat([df, temp_splines, rh_splines], axis=1)
    
    spline_info = {
        'n_knots': n_knots,
        'df': df_splines,
        'temp_scaler': scaler_temp,
        'rh_scaler': scaler_rh,
        'temp_formula': temp_formula,
        'rh_formula': rh_formula
    }
    
    print(f"\nCreated spline features with {n_knots} knots:")
    print(f"  Temperature: {len(temp_splines.columns)} basis functions (1 dropped to remove constant)")
    print(f"  Relative Humidity: {len(rh_splines.columns)} basis functions (1 dropped to remove constant)")
    print(f"  Total: {len(temp_splines.columns) + len(rh_splines.columns)} spline features")
    
    return df, spline_info


def create_lag_features(df: pd.DataFrame, 
                       oreo_col: str = 'oreo_calls',
                       gastro_col: str = 'gastro_calls',
                       max_lag: int = 1) -> pd.DataFrame:
    """
    Create lag features for serial dependence and cross-species influence.
    
    Args:
        df: Input DataFrame
        oreo_col: Oreobates calls column
        gastro_col: Gastrotheca calls column  
        max_lag: Maximum lag to include (default 1 hour)
        
    Returns:
        DataFrame with lag features
    """
    print("\n" + "=" * 80)
    print("STEP 4: BUILDING LAG FEATURES (Serial & Cross-Species)")
    print("=" * 80)
    
    df = df.copy()
    
    # Create lags for both species
    for lag in range(1, max_lag + 1):
        df[f'oreo_lag{lag}'] = df[oreo_col].shift(lag)
        df[f'gastro_lag{lag}'] = df[gastro_col].shift(lag)
    
    print(f"\nCreated lag features (lag 1 to {max_lag}):")
    print(f"  Oreobates lags: {max_lag} features")
    print(f"  Gastrotheca lags: {max_lag} features")
    print(f"  Total: {max_lag * 2} lag features")
    
    # Remove rows with NaN from lagging
    df = df.dropna()
    print(f"\nRemoved {max_lag} initial rows with missing lag values")
    print(f"Final dataset: {len(df)} observations")
    
    return df


def fit_ordinal_model(df: pd.DataFrame,
                     response_col: str,
                     predictor_cols: List[str],
                     model_name: str = "Model",
                     distr: str = "logit") -> Tuple:
    """
    Fit ordinal logistic regression model.
    
    Args:
        df: Input DataFrame
        response_col: Name of response variable (ordinal)
        predictor_cols: List of predictor column names
        model_name: Name for the model (for display)
        distr: Distribution ('logit' or 'probit')
        
    Returns:
        Tuple of (fitted_result, model_object, valid_indices)
    """
    print(f"\n" + "-" * 80)
    print(f"Fitting {model_name}")
    print("-" * 80)
    
    # Prepare data
    y = df[response_col].astype(int)
    X = df[predictor_cols].copy()
    
    # Check for missing values
    valid_mask = ~(y.isna() | X.isna().any(axis=1))
    y_valid = y[valid_mask]
    X_valid = X[valid_mask]
    
    # Check for and remove constant columns
    constant_cols = []
    for col in X_valid.columns:
        if X_valid[col].std() < 1e-10:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        X_valid = X_valid.drop(columns=constant_cols)
        predictor_cols = [col for col in predictor_cols if col not in constant_cols]
    
    print(f"Response: {response_col}")
    print(f"Predictors: {len(X_valid.columns)} variables")
    print(f"Observations: {len(y_valid)}")
    print(f"Categories: {sorted(y_valid.unique())}")
    
    # Fit model
    model = OrderedModel(y_valid, X_valid, distr=distr)
    result = model.fit(method='bfgs', maxiter=1000, disp=False)
    
    print(f"\nModel converged: {result.mle_retvals['converged']}")
    print(f"Log-likelihood: {result.llf:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    
    return result, model, valid_mask


def interpret_model_coefficients(result, species_name: str) -> pd.DataFrame:
    """
    Extract and interpret model coefficients.
    
    Args:
        result: Fitted model result
        species_name: Name of species being modeled
        
    Returns:
        DataFrame with coefficient estimates and interpretations
    """
    print(f"\n{'=' * 80}")
    print(f"COEFFICIENT INTERPRETATION: {species_name}")
    print(f"{'=' * 80}")
    
    # Extract coefficients (exclude threshold parameters)
    # Thresholds are at the end of params and have '/' in the name
    all_param_names = result.params.index.tolist()
    predictor_cols = [name for name in all_param_names if '/' not in name]
    n_predictors = len(predictor_cols)
    
    # Get indices of predictor parameters
    pred_indices = [i for i, name in enumerate(all_param_names) if '/' not in name]
    
    coefs = result.params.iloc[pred_indices]
    se = result.bse.iloc[pred_indices]
    pvals = result.pvalues.iloc[pred_indices]
    
    # Create interpretation dataframe
    coef_df = pd.DataFrame({
        'Variable': predictor_cols,
        'Coefficient': coefs.values,
        'Std.Error': se.values,
        'z-value': coefs.values / se.values,
        'P>|z|': pvals.values,
        'Odds Ratio': np.exp(coefs.values)
    })
    
    # Add significance stars
    def sig_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        elif p < 0.1:
            return '.'
        else:
            return ''
    
    coef_df['Sig'] = coef_df['P>|z|'].apply(sig_stars)
    
    # Sort by absolute coefficient value
    coef_df['abs_coef'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # Print top effects
    print("\nTop 15 Effects (by magnitude):")
    print("-" * 80)
    display_df = coef_df[['Variable', 'Coefficient', 'Std.Error', 'P>|z|', 'Odds Ratio', 'Sig']].head(15)
    print(display_df.to_string(index=False))
    
    # Interpret key effects
    print("\n\nKEY INTERPRETATIONS:")
    print("-" * 80)
    
    # Find lag effects
    own_lags = coef_df[coef_df['Variable'].str.contains('lag') & 
                      ~coef_df['Variable'].str.contains('gastro' if 'oreo' in species_name.lower() else 'oreo')]
    cross_lags = coef_df[coef_df['Variable'].str.contains('lag') & 
                        coef_df['Variable'].str.contains('gastro' if 'oreo' in species_name.lower() else 'oreo')]
    
    if len(own_lags) > 0:
        print("\n1. SERIAL DEPENDENCE (Own Species Lags):")
        for _, row in own_lags.iterrows():
            if row['P>|z|'] < 0.05:
                direction = "increases" if row['Coefficient'] > 0 else "decreases"
                print(f"   • {row['Variable']}: Coefficient = {row['Coefficient']:.3f} {row['Sig']}")
                print(f"     Previous hour's calls {direction} probability of higher intensity")
                print(f"     Odds ratio: {row['Odds Ratio']:.2f}")
    
    if len(cross_lags) > 0:
        print("\n2. CROSS-SPECIES INFLUENCE:")
        for _, row in cross_lags.iterrows():
            if row['P>|z|'] < 0.05:
                direction = "increases" if row['Coefficient'] > 0 else "decreases"
                other_species = "Gastrotheca" if 'oreo' in species_name.lower() else "Oreobates"
                print(f"   • {row['Variable']}: Coefficient = {row['Coefficient']:.3f} {row['Sig']}")
                print(f"     {other_species} calls {direction} probability of higher intensity")
                print(f"     Odds ratio: {row['Odds Ratio']:.2f}")
    
    # Find environmental effects
    temp_effects = coef_df[coef_df['Variable'].str.contains('temp')]
    rh_effects = coef_df[coef_df['Variable'].str.contains('rh')]
    
    print("\n3. ENVIRONMENTAL EFFECTS:")
    print("   Temperature and RH effects are captured by splines (see plots)")
    print(f"   • Temperature: {len(temp_effects)} spline basis functions")
    print(f"   • Relative Humidity: {len(rh_effects)} spline basis functions")
    
    # Cyclic effects
    hour_effects = coef_df[coef_df['Variable'].str.contains('hour_')]
    doy_effects = coef_df[coef_df['Variable'].str.contains('doy_')]
    
    print("\n4. TEMPORAL PATTERNS:")
    print(f"   • Hour-of-day: {len(hour_effects)} Fourier terms")
    print(f"   • Day-of-year: {len(doy_effects)} Fourier terms")
    print("   (See time series plots for diurnal/seasonal patterns)")
    
    return coef_df.drop('abs_coef', axis=1)


def predict_probabilities_grid(result, df: pd.DataFrame, 
                               predictor_cols: List[str],
                               spline_info: Dict,
                               temp_range: Tuple[float, float],
                               rh_range: Tuple[float, float],
                               fixed_hour: int = 22,
                               fixed_doy: int = 180,
                               fixed_lags: Dict = None,
                               grid_size: int = 50) -> Tuple:
    """
    Generate predicted probabilities on a temperature-humidity grid.
    
    Args:
        result: Fitted model result
        df: DataFrame with spline columns for reference
        predictor_cols: List of predictor names
        spline_info: Dictionary with spline information
        temp_range: (min, max) temperature range
        rh_range: (min, max) RH range
        fixed_hour: Hour of day to fix (0-23)
        fixed_doy: Day of year to fix (1-365)
        fixed_lags: Dictionary of lag values to fix
        grid_size: Number of points in each dimension
        
    Returns:
        Tuple of (temp_grid, rh_grid, prob_grid_dict)
    """
    # Create grid
    temp_vals = np.linspace(temp_range[0], temp_range[1], grid_size)
    rh_vals = np.linspace(rh_range[0], rh_range[1], grid_size)
    temp_grid, rh_grid = np.meshgrid(temp_vals, rh_vals)
    
    # Flatten for prediction
    temp_flat = temp_grid.flatten()
    rh_flat = rh_grid.flatten()
    
    # Create prediction dataframe
    pred_df = pd.DataFrame()
    
    # Add temperature and RH
    pred_df['temp'] = temp_flat
    pred_df['rh'] = rh_flat
    
    # Standardize
    pred_df['temp_std'] = spline_info['temp_scaler'].transform(pred_df[['temp']])
    pred_df['rh_std'] = spline_info['rh_scaler'].transform(pred_df[['rh']])
    
    # Add splines
    temp_formula = spline_info['temp_formula']
    rh_formula = spline_info['rh_formula']
    
    temp_splines = dmatrix(temp_formula, pred_df, return_type='dataframe')
    temp_splines.columns = [f'temp_spline_{i}' for i in range(len(temp_splines.columns))]
    # Drop first column to match training (natural splines sum to constant)
    temp_splines = temp_splines.iloc[:, 1:]
    
    rh_splines = dmatrix(rh_formula, pred_df, return_type='dataframe')
    rh_splines.columns = [f'rh_spline_{i}' for i in range(len(rh_splines.columns))]
    # Drop first column to match training
    rh_splines = rh_splines.iloc[:, 1:]
    
    pred_df = pd.concat([pred_df, temp_splines, rh_splines], axis=1)
    
    # Add Fourier features
    hour = fixed_hour
    for k in range(1, 3):
        pred_df[f'hour_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24)
        pred_df[f'hour_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24)
    
    doy = fixed_doy
    for k in range(1, 4):
        pred_df[f'doy_sin_{k}'] = np.sin(2 * np.pi * k * doy / 365)
        pred_df[f'doy_cos_{k}'] = np.cos(2 * np.pi * k * doy / 365)
    
    # Add lag features (use provided values or 0)
    if fixed_lags is None:
        fixed_lags = {}
    
    for col in predictor_cols:
        if 'lag' in col:
            pred_df[col] = fixed_lags.get(col, 0)
    
    # Ensure all predictor columns are present
    for col in predictor_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
    
    # Reorder columns to match model
    X_pred = pred_df[predictor_cols]
    
    # Predict probabilities
    probs = result.predict(exog=X_pred)
    
    # Reshape to grid
    prob_grid_dict = {}
    categories = np.unique(result.model.endog)
    for i, cat in enumerate(sorted(categories)):
        prob_grid_dict[cat] = probs.iloc[:, i].values.reshape(grid_size, grid_size)
    
    return temp_grid, rh_grid, prob_grid_dict


def plot_comprehensive_results(df: pd.DataFrame,
                               oreo_result, gastro_result,
                               spline_info: Dict,
                               save_path: str = None):
    """
    Create comprehensive visualization of model results.
    
    Args:
        df: DataFrame with all features
        oreo_result: Fitted Oreobates model result
        gastro_result: Fitted Gastrotheca model result
        spline_info: Spline information dictionary
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE PLOTS")
    print("=" * 80)
    
    # Get predictor columns from results
    oreo_pred_cols = [col for col in oreo_result.model.exog_names if '/' not in col]  # Exclude thresholds
    gastro_pred_cols = [col for col in gastro_result.model.exog_names if '/' not in col]  # Exclude thresholds
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get predictions for time series
    X_oreo = df[oreo_pred_cols]
    X_gastro = df[gastro_pred_cols]
    
    oreo_probs = oreo_result.predict(exog=X_oreo)
    gastro_probs = gastro_result.predict(exog=X_gastro)
    
    # Expected values
    oreo_cats = sorted(df['oreo_calls'].unique())
    gastro_cats = sorted(df['gastro_calls'].unique())
    
    oreo_expected = (oreo_probs * oreo_cats).sum(axis=1)
    gastro_expected = (gastro_probs * gastro_cats).sum(axis=1)
    
    # 1. Time series of predictions vs actual (Oreobates)
    ax1 = fig.add_subplot(gs[0, :2])
    sample_idx = df.index[:500]  # First 500 hours
    ax1.plot(sample_idx, df.loc[sample_idx, 'oreo_calls'], 
             'o', alpha=0.3, markersize=3, label='Actual', color='black')
    ax1.plot(sample_idx, oreo_expected.loc[sample_idx], 
             '-', alpha=0.8, linewidth=2, label='Predicted (Expected)', color='C0')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Call Intensity')
    ax1.set_title('Oreobates berdemenos: Model Predictions vs Actual', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series of predictions vs actual (Gastrotheca)
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(sample_idx, df.loc[sample_idx, 'gastro_calls'], 
             'o', alpha=0.3, markersize=3, label='Actual', color='black')
    ax2.plot(sample_idx, gastro_expected.loc[sample_idx], 
             '-', alpha=0.8, linewidth=2, label='Predicted (Expected)', color='C1')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Call Intensity')
    ax2.set_title('Gastrotheca chysosticta: Model Predictions vs Actual', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals (Oreobates)
    ax3 = fig.add_subplot(gs[0, 2])
    oreo_resid = df['oreo_calls'] - oreo_expected
    ax3.hist(oreo_resid, bins=30, alpha=0.7, color='C0', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Oreobates Residuals\nMAE={np.abs(oreo_resid).mean():.3f}', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals (Gastrotheca)
    ax4 = fig.add_subplot(gs[1, 2])
    gastro_resid = df['gastro_calls'] - gastro_expected
    ax4.hist(gastro_resid, bins=30, alpha=0.7, color='C1', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Gastrotheca Residuals\nMAE={np.abs(gastro_resid).mean():.3f}', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Probability heatmap - Oreobates (high intensity, category 3)
    ax5 = fig.add_subplot(gs[2, 0])
    temp_range = (df['temp'].min(), df['temp'].max())
    rh_range = (df['rh'].min(), df['rh'].max())
    
    temp_grid, rh_grid, oreo_prob_grid = predict_probabilities_grid(
        oreo_result, df, oreo_pred_cols, spline_info,
        temp_range, rh_range, fixed_hour=22, fixed_doy=180,
        fixed_lags={'oreo_lag1': 1, 'gastro_lag1': 0}, grid_size=40
    )
    
    # Plot highest category probability
    max_cat = max(oreo_prob_grid.keys())
    im5 = ax5.contourf(temp_grid, rh_grid, oreo_prob_grid[max_cat], 
                       levels=15, cmap='YlOrRd')
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('Relative Humidity (%)')
    ax5.set_title(f'Oreobates: P(Intensity={max_cat})\nat 22:00, mid-year', fontsize=10)
    plt.colorbar(im5, ax=ax5, label='Probability')
    
    # 6. Probability heatmap - Gastrotheca (high intensity)
    ax6 = fig.add_subplot(gs[2, 1])
    temp_grid, rh_grid, gastro_prob_grid = predict_probabilities_grid(
        gastro_result, df, gastro_pred_cols, spline_info,
        temp_range, rh_range, fixed_hour=22, fixed_doy=180,
        fixed_lags={'oreo_lag1': 0, 'gastro_lag1': 1}, grid_size=40
    )
    
    max_cat = max(gastro_prob_grid.keys())
    im6 = ax6.contourf(temp_grid, rh_grid, gastro_prob_grid[max_cat], 
                       levels=15, cmap='YlGnBu')
    ax6.set_xlabel('Temperature (°C)')
    ax6.set_ylabel('Relative Humidity (%)')
    ax6.set_title(f'Gastrotheca: P(Intensity={max_cat})\nat 22:00, mid-year', fontsize=10)
    plt.colorbar(im6, ax=ax6, label='Probability')
    
    # 7. Difference in probabilities (interaction effect)
    ax7 = fig.add_subplot(gs[2, 2])
    # Difference when other species is active vs not
    _, _, oreo_prob_grid_noG = predict_probabilities_grid(
        oreo_result, df, oreo_pred_cols, spline_info,
        temp_range, rh_range, fixed_hour=22, fixed_doy=180,
        fixed_lags={'oreo_lag1': 1, 'gastro_lag1': 0}, grid_size=40
    )
    _, _, oreo_prob_grid_withG = predict_probabilities_grid(
        oreo_result, df, oreo_pred_cols, spline_info,
        temp_range, rh_range, fixed_hour=22, fixed_doy=180,
        fixed_lags={'oreo_lag1': 1, 'gastro_lag1': 2}, grid_size=40
    )
    
    max_cat = max(oreo_prob_grid_noG.keys())
    prob_diff = oreo_prob_grid_withG[max_cat] - oreo_prob_grid_noG[max_cat]
    
    im7 = ax7.contourf(temp_grid, rh_grid, prob_diff, 
                       levels=15, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax7.set_xlabel('Temperature (°C)')
    ax7.set_ylabel('Relative Humidity (%)')
    ax7.set_title(f'Cross-Species Effect\nΔP(Oreo={max_cat}) when Gastro active', fontsize=10)
    plt.colorbar(im7, ax=ax7, label='Probability Change')
    
    # 8. Diurnal patterns
    ax8 = fig.add_subplot(gs[3, 0])
    hours = df.index.hour
    oreo_by_hour = df.groupby(hours)['oreo_calls'].mean()
    gastro_by_hour = df.groupby(hours)['gastro_calls'].mean()
    
    ax8.plot(oreo_by_hour.index, oreo_by_hour.values, 'o-', 
             linewidth=2, markersize=6, label='Oreobates', color='C0')
    ax8.plot(gastro_by_hour.index, gastro_by_hour.values, 's-', 
             linewidth=2, markersize=6, label='Gastrotheca', color='C1')
    ax8.set_xlabel('Hour of Day')
    ax8.set_ylabel('Mean Call Intensity')
    ax8.set_title('Diurnal Pattern', fontsize=10, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(range(0, 24, 3))
    
    # 9. Seasonal patterns
    ax9 = fig.add_subplot(gs[3, 1])
    doy = df.index.dayofyear
    oreo_by_doy = df.groupby(doy)['oreo_calls'].mean()
    gastro_by_doy = df.groupby(doy)['gastro_calls'].mean()
    
    # Smooth for visualization
    from scipy.ndimage import gaussian_filter1d
    oreo_smooth = gaussian_filter1d(oreo_by_doy.values, sigma=5)
    gastro_smooth = gaussian_filter1d(gastro_by_doy.values, sigma=5)
    
    ax9.plot(oreo_by_doy.index, oreo_smooth, '-', 
             linewidth=2, label='Oreobates', color='C0', alpha=0.8)
    ax9.plot(gastro_by_doy.index, gastro_smooth, '-', 
             linewidth=2, label='Gastrotheca', color='C1', alpha=0.8)
    ax9.set_xlabel('Day of Year')
    ax9.set_ylabel('Mean Call Intensity')
    ax9.set_title('Seasonal Pattern (smoothed)', fontsize=10, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Model comparison metrics
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    
    metrics_text = f"""
    MODEL PERFORMANCE METRICS
    {'=' * 35}
    
    OREOBATES BERDEMENOS
    AIC: {oreo_result.aic:.1f}
    BIC: {oreo_result.bic:.1f}
    Log-Lik: {oreo_result.llf:.1f}
    MAE: {np.abs(oreo_resid).mean():.3f}
    RMSE: {np.sqrt((oreo_resid**2).mean()):.3f}
    
    GASTROTHECA CHYSOSTICTA
    AIC: {gastro_result.aic:.1f}
    BIC: {gastro_result.bic:.1f}
    Log-Lik: {gastro_result.llf:.1f}
    MAE: {np.abs(gastro_resid).mean():.3f}
    RMSE: {np.sqrt((gastro_resid**2).mean()):.3f}
    """
    
    ax10.text(0.1, 0.5, metrics_text, fontsize=10, 
              family='monospace', verticalalignment='center')
    
    plt.suptitle('Ordinal Regression Analysis: Amphibian Call Intensities', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


def assess_validity(oreo_result, gastro_result, df: pd.DataFrame):
    """
    Assess the validity of model inferences.
    
    Args:
        oreo_result: Fitted Oreobates model
        gastro_result: Fitted Gastrotheca model
        df: DataFrame with data
    """
    print("\n" + "=" * 80)
    print("VALIDITY ASSESSMENT OF INFERENCES")
    print("=" * 80)
    
    # Get predictor columns from results (exclude thresholds)
    oreo_pred_cols = [col for col in oreo_result.model.exog_names if '/' not in col]
    gastro_pred_cols = [col for col in gastro_result.model.exog_names if '/' not in col]
    
    print("\n1. MODEL CONVERGENCE")
    print("-" * 80)
    print(f"   Oreobates model converged: {oreo_result.mle_retvals['converged']}")
    print(f"   Gastrotheca model converged: {gastro_result.mle_retvals['converged']}")
    
    print("\n2. PROPORTIONAL ODDS ASSUMPTION")
    print("-" * 80)
    print("   The ordinal logistic model assumes proportional odds across categories.")
    print("   This means the effect of predictors is constant across different thresholds.")
    print("   Violation would suggest using partial proportional odds or multinomial models.")
    print("   → Check: Compare AIC/BIC with multinomial logit if concerned")
    
    print("\n3. SERIAL CORRELATION IN RESIDUALS")
    print("-" * 80)
    X_oreo = df[oreo_pred_cols]
    X_gastro = df[gastro_pred_cols]
    
    oreo_probs = oreo_result.predict(exog=X_oreo)
    gastro_probs = gastro_result.predict(exog=X_gastro)
    
    oreo_cats = sorted(df['oreo_calls'].unique())
    gastro_cats = sorted(df['gastro_calls'].unique())
    
    oreo_expected = (oreo_probs * oreo_cats).sum(axis=1)
    gastro_expected = (gastro_probs * gastro_cats).sum(axis=1)
    
    oreo_resid = df['oreo_calls'] - oreo_expected
    gastro_resid = df['gastro_calls'] - gastro_expected
    
    # Check autocorrelation
    from scipy.stats import pearsonr
    
    oreo_acf1 = pearsonr(oreo_resid[:-1], oreo_resid[1:])[0]
    gastro_acf1 = pearsonr(gastro_resid[:-1], gastro_resid[1:])[0]
    
    print(f"   Oreobates residual ACF(1): {oreo_acf1:.3f}")
    print(f"   Gastrotheca residual ACF(1): {gastro_acf1:.3f}")
    print("   → Lag-1 predictors should capture most autocorrelation")
    print(f"   → {'✓ Low residual autocorrelation' if abs(oreo_acf1) < 0.1 and abs(gastro_acf1) < 0.1 else '⚠ Some residual autocorrelation remains'}")
    
    print("\n4. MULTICOLLINEARITY")
    print("-" * 80)
    from numpy.linalg import cond
    
    oreo_cond = cond(df[oreo_pred_cols].values)
    gastro_cond = cond(df[gastro_pred_cols].values)
    
    print(f"   Oreobates design matrix condition number: {oreo_cond:.1f}")
    print(f"   Gastrotheca design matrix condition number: {gastro_cond:.1f}")
    print("   → Condition number < 30: Good")
    print("   → Condition number 30-1000: Moderate multicollinearity")
    print("   → Condition number > 1000: Severe multicollinearity")
    
    print("\n5. SAMPLE SIZE ADEQUACY")
    print("-" * 80)
    n_obs = len(df)
    n_params_oreo = len(oreo_result.params)
    n_params_gastro = len(gastro_result.params)
    
    print(f"   Observations: {n_obs}")
    print(f"   Oreobates parameters: {n_params_oreo} (ratio: {n_obs/n_params_oreo:.1f}:1)")
    print(f"   Gastrotheca parameters: {n_params_gastro} (ratio: {n_obs/n_params_gastro:.1f}:1)")
    print("   → Rule of thumb: At least 10-20 observations per parameter")
    print(f"   → {'✓ Adequate sample size' if n_obs/max(n_params_oreo, n_params_gastro) > 10 else '⚠ Consider simplifying model'}")
    
    print("\n6. CROSS-SPECIES EFFECTS SIGNIFICANCE")
    print("-" * 80)
    
    # Extract cross-species lag coefficients
    oreo_cross_idx = [i for i, col in enumerate(oreo_pred_cols) if 'gastro_lag' in col]
    gastro_cross_idx = [i for i, col in enumerate(gastro_pred_cols) if 'oreo_lag' in col]
    
    if oreo_cross_idx:
        oreo_cross_pval = oreo_result.pvalues[oreo_cross_idx[0]]
        print(f"   Gastrotheca → Oreobates effect p-value: {oreo_cross_pval:.4f}")
        print(f"   → {'✓ Significant' if oreo_cross_pval < 0.05 else '✗ Not significant'}")
    
    if gastro_cross_idx:
        gastro_cross_pval = gastro_result.pvalues[gastro_cross_idx[0]]
        print(f"   Oreobates → Gastrotheca effect p-value: {gastro_cross_pval:.4f}")
        print(f"   → {'✓ Significant' if gastro_cross_pval < 0.05 else '✗ Not significant'}")
    
    print("\n7. OUT-OF-SAMPLE VALIDATION")
    print("-" * 80)
    print("   Recommendation: Perform k-fold cross-validation or time-series split")
    print("   to assess generalization performance on held-out data.")
    
    print("\n8. CAUSAL INTERPRETATION CAVEATS")
    print("-" * 80)
    print("   ⚠ IMPORTANT: These models show ASSOCIATION, not CAUSATION")
    print("   • Cross-species 'influence' could reflect:")
    print("     - Shared environmental drivers not fully captured")
    print("     - Common behavioral responses to unmeasured factors")
    print("     - True behavioral interaction")
    print("   • Temporal precedence (lags) suggests potential causality but doesn't prove it")
    print("   • Experimental manipulation would be needed for causal claims")
    
    print("\n" + "=" * 80)


def main():
    """
    Main execution function for end-to-end ordinal regression analysis.
    """
    print("\n" + "=" * 80)
    print("ORDINAL REGRESSION ANALYSIS FOR AMPHIBIAN CALL INTENSITIES")
    print("=" * 80)
    print("\nModeling framework:")
    print("  • Cumulative-link (ordinal) regression")
    print("  • Splines for nonlinear environmental effects")
    print("  • Fourier terms for cyclic temporal patterns")
    print("  • Lags for serial dependence and cross-species influence")
    print("=" * 80)
    
    # File path
    filepath = 'sample_helechos_data.csv'
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data(filepath)
    
    # Step 2: Create Fourier features
    df = create_fourier_features(df)
    
    # Step 3: Create spline features
    df, spline_info = create_spline_features(df, n_knots=5)
    
    # Step 4: Create lag features
    df = create_lag_features(df, max_lag=1)
    
    # Step 5: Define predictors
    # Spline columns
    temp_spline_cols = [col for col in df.columns if 'temp_spline' in col]
    rh_spline_cols = [col for col in df.columns if 'rh_spline' in col]
    
    # Fourier columns
    hour_fourier_cols = [col for col in df.columns if 'hour_sin' in col or 'hour_cos' in col]
    doy_fourier_cols = [col for col in df.columns if 'doy_sin' in col or 'doy_cos' in col]
    
    # Oreobates model predictors
    oreo_predictors = (
        ['oreo_lag1', 'gastro_lag1'] +  # Lags
        temp_spline_cols + rh_spline_cols +  # Splines
        hour_fourier_cols + doy_fourier_cols  # Fourier
    )
    
    # Gastrotheca model predictors
    gastro_predictors = (
        ['gastro_lag1', 'oreo_lag1'] +  # Lags
        temp_spline_cols + rh_spline_cols +  # Splines
        hour_fourier_cols + doy_fourier_cols  # Fourier
    )
    
    print("\n" + "=" * 80)
    print("STEP 5: FITTING ORDINAL MODELS")
    print("=" * 80)
    
    # Step 6: Fit Oreobates model
    oreo_result, oreo_model, oreo_mask = fit_ordinal_model(
        df, 'oreo_calls', oreo_predictors,
        model_name="Oreobates berdemenos Model",
        distr='logit'
    )
    
    # Step 7: Fit Gastrotheca model
    gastro_result, gastro_model, gastro_mask = fit_ordinal_model(
        df, 'gastro_calls', gastro_predictors,
        model_name="Gastrotheca chysosticta Model",
        distr='logit'
    )
    
    # Step 8: Interpret coefficients
    oreo_coefs = interpret_model_coefficients(
        oreo_result, "Oreobates berdemenos"
    )
    
    gastro_coefs = interpret_model_coefficients(
        gastro_result, "Gastrotheca chysosticta"
    )
    
    # Step 9: Create comprehensive plots
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_comprehensive_results(
        df, oreo_result, gastro_result,
        spline_info,
        save_path='plots/ordinal_regression_comprehensive.png'
    )
    
    # Step 10: Assess validity
    assess_validity(oreo_result, gastro_result, df)
    
    # Step 11: Generate example predictions
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    print("\nNote: Detailed prediction grids are shown in the comprehensive plot.")
    print("The plot includes:")
    print("  • Time series of model predictions vs actual data")
    print("  • Probability heatmaps over Temperature-Humidity space")
    print("  • Cross-species interaction effects")
    print("  • Diurnal and seasonal patterns")
    
    # Simple example using actual data points
    print("\nExample predictions from the dataset:")
    print("-" * 80)
    
    # Get predictor columns from results (exclude thresholds)
    oreo_predictors = [col for col in oreo_result.model.exog_names if '/' not in col]
    gastro_predictors = [col for col in gastro_result.model.exog_names if '/' not in col]
    
    # Select a few interesting examples from the data
    example_indices = [100, 500, 1000, 1500, 2000]
    
    for idx in example_indices:
        if idx < len(df):
            row = df.iloc[idx]
            
            # Get predictors for this row
            X_oreo_row = df.loc[[df.index[idx]], oreo_predictors]
            X_gastro_row = df.loc[[df.index[idx]], gastro_predictors]
            
            # Get predictions
            oreo_prob = oreo_result.predict(exog=X_oreo_row)
            gastro_prob = gastro_result.predict(exog=X_gastro_row)
            
            print(f"\nTime: {df.index[idx]}")
            print(f"  Temp: {row['temp']:.1f}°C, RH: {row['rh']:.1f}%")
            print(f"  Actual: Oreo={row['oreo_calls']}, Gastro={row['gastro_calls']}")
            print(f"  Oreobates probabilities: ", end="")
            for i in range(4):
                print(f"P({i})={oreo_prob.iloc[0, i]:.3f} ", end="")
            print(f"\n  Gastrotheca probabilities: ", end="")
            for i in range(4):
                print(f"P({i})={gastro_prob.iloc[0, i]:.3f} ", end="")
            print()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print("  • Model summaries and coefficient interpretations (above)")
    print("  • Comprehensive visualization: plots/ordinal_regression_comprehensive.png")
    print("  • Validity assessment and caveats (above)")
    print("\nKey findings:")
    print("  1. Both models account for temporal autocorrelation via lags")
    print("  2. Splines capture nonlinear environmental effects")
    print("  3. Fourier terms model diurnal and seasonal cycles")
    print("  4. Cross-species lags test for behavioral association")
    print("  5. Predicted probabilities show how call intensity varies with conditions")
    print("\nRemember: Association ≠ Causation! See validity assessment above.")
    print("=" * 80)


if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    main()
