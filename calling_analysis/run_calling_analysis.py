#!/usr/bin/env python3
"""
Reproduce the paper's core calling analyses for a two-species dataset.

This script loads hourly call-index data, derives presence/absence and richness,
identifies complete days, computes daily calling effort, fits GLM models with
day fixed effects, performs AIC-based model selection, and produces plots,
tables, and a concise markdown summary report.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: Path
    output_dir: Path
    datetime_col: str = "DateTime"
    temp_col: str = "Temp"
    humidity_col: str = "RH%"
    species1_col: str = "Gastrotheca chrysosticta"
    species2_col: str = "Oreobates berdemenos"
    rain_col: str = "Rain"
    hour_as_categorical: bool = True
    collinearity_threshold: float = 0.7


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    subdirs = {
        "data": output_dir / "data",
        "tables": output_dir / "tables",
        "plots": output_dir / "plots",
        "report": output_dir / "report",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def load_data(path: Path, datetime_col: str) -> pd.DataFrame:
    LOGGER.info("Loading data from %s", path)
    df = pd.read_csv(path)
    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col)
    return df


def prepare_data(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    df = df.copy()
    required = [config.species1_col, config.species2_col, config.temp_col, config.humidity_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dt = pd.to_datetime(df[config.datetime_col], errors="coerce")
    df = df.assign(
        date=dt.dt.date,
        hour=dt.dt.hour,
        day_id=dt.dt.date.astype(str),
    )

    for col in [config.temp_col, config.humidity_col, config.species1_col, config.species2_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["species1_presence"] = np.where(
        df[config.species1_col].isna(),
        np.nan,
        (df[config.species1_col] > 0).astype(int),
    )
    df["species2_presence"] = np.where(
        df[config.species2_col].isna(),
        np.nan,
        (df[config.species2_col] > 0).astype(int),
    )
    df["any_calling"] = df[["species1_presence", "species2_presence"]].max(axis=1)
    df["richness"] = df[["species1_presence", "species2_presence"]].sum(axis=1, min_count=2)
    df["species1_abundance"] = df[config.species1_col]
    df["species2_abundance"] = df[config.species2_col]
    df["other_species_calling_for_species1"] = df["species2_presence"]
    df["other_species_calling_for_species2"] = df["species1_presence"]
    df["day_period"] = np.where((df["hour"] >= 20) | (df["hour"] <= 5), "night", "day")

    if config.rain_col not in df.columns:
        LOGGER.warning("Rain column '%s' not found; skipping rain-related checks.", config.rain_col)

    return df


def data_quality_checks(df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, pd.DataFrame]:
    missing = df.isna().sum().to_frame(name="missing_count")
    dup_count = df.duplicated(subset=[config.datetime_col]).sum()
    obs_by_day = df.groupby("day_id").size().rename("n_obs").to_frame()
    complete_days = obs_by_day[obs_by_day["n_obs"] == 24].shape[0]
    incomplete_days = obs_by_day[obs_by_day["n_obs"] != 24].shape[0]
    summary = pd.DataFrame(
        {
            "duplicate_timestamps": [dup_count],
            "complete_days": [complete_days],
            "incomplete_days": [incomplete_days],
            "total_days": [obs_by_day.shape[0]],
        }
    )
    return {"missing": missing, "obs_by_day": obs_by_day, "summary": summary}


def get_complete_days(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("day_id").size()
    complete_day_ids = counts[counts == 24].index
    return df[df["day_id"].isin(complete_day_ids)].copy()


def compute_daily_effort(complete_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        complete_df.groupby("day_id")[["species1_presence", "species2_presence"]]
        .sum(min_count=24)
        .rename(
            columns={
                "species1_presence": "species1_daily_calling_effort",
                "species2_presence": "species2_daily_calling_effort",
            }
        )
    )
    daily = daily.reset_index()
    return daily


def compute_temp_humidity_correlation(df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, float]:
    subset = df[[config.temp_col, config.humidity_col]].dropna()
    if subset.empty:
        return {"r": np.nan, "p_value": np.nan, "n": 0}
    if subset[config.temp_col].nunique() < 2 or subset[config.humidity_col].nunique() < 2:
        return {"r": np.nan, "p_value": np.nan, "n": subset.shape[0]}
    r, p = stats.pearsonr(subset[config.temp_col], subset[config.humidity_col])
    return {"r": r, "p_value": p, "n": subset.shape[0]}


def summarize_circadian(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    grouped = df.groupby("hour")
    summary = pd.DataFrame(
        {
            "mean_temperature": grouped[config.temp_col].mean(),
            "mean_humidity": grouped[config.humidity_col].mean(),
            "mean_richness": grouped["richness"].mean(),
            "prop_any_calling": grouped["any_calling"].mean(),
            "prop_species1_calling": grouped["species1_presence"].mean(),
            "prop_species2_calling": grouped["species2_presence"].mean(),
        }
    ).reset_index()
    return summary


def summarize_day_night(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("day_period")
    return grouped[["any_calling", "species1_presence", "species2_presence", "richness"]].mean()


def build_formula(
    response: str,
    predictors: Iterable[str],
    day_id_col: str = "day_id",
    hour_as_categorical: bool = True,
) -> str:
    def quote_name(name: str) -> str:
        if name.isidentifier():
            return name
        escaped = name.replace('"', '\\"')
        return f'Q("{escaped}")'

    terms: List[str] = []
    for pred in predictors:
        if pred == "hour" and hour_as_categorical:
            terms.append("C(hour)")
        else:
            terms.append(quote_name(pred))
    terms.append(f"C({quote_name(day_id_col)})")
    if terms:
        return f"{response} ~ " + " + ".join(terms)
    return f"{response} ~ C({day_id_col})"


def fit_glm_model(formula: str, df: pd.DataFrame, family: sm.families.Family):
    model = smf.glm(formula=formula, data=df, family=family)
    result = model.fit()
    return result


def fit_candidate_models(
    df: pd.DataFrame,
    response: str,
    predictors_list: List[Tuple[str, List[str]]],
    family: sm.families.Family,
    hour_as_categorical: bool = True,
) -> Dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    results = {}
    for name, predictors in predictors_list:
        formula = build_formula(response, predictors, hour_as_categorical=hour_as_categorical)
        results[name] = fit_glm_model(formula, df, family)
    return results


def model_selection_table(results: Dict[str, sm.regression.linear_model.RegressionResultsWrapper]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({"model": name, "aic": res.aic, "nobs": res.nobs})
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def coefficient_table(res: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    conf = res.conf_int()
    table = pd.DataFrame(
        {
            "term": res.params.index,
            "estimate": res.params.values,
            "std_error": res.bse.values,
            "z_value": res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
        }
    )
    return table


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_environmental_circadian(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(summary["hour"], summary["mean_temperature"], color="tab:red", label="Temperature")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Mean temperature", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(summary["hour"], summary["mean_humidity"], color="tab:blue", label="Humidity")
    ax2.set_ylabel("Mean humidity", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Mean temperature and humidity by hour")
    save_plot(fig, output_dir / "environmental_circadian")


def plot_environmental_timeseries(df: pd.DataFrame, config: AnalysisConfig, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df[config.datetime_col], df[config.temp_col], color="tab:red", label="Temperature")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(df[config.datetime_col], df[config.humidity_col], color="tab:blue", alpha=0.6, label="Humidity")
    ax2.set_ylabel("Humidity", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Environmental time series")
    save_plot(fig, output_dir / "environmental_timeseries")


def plot_hourly_richness(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary["hour"], summary["mean_richness"], color="tab:green")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean richness")
    ax.set_title("Mean richness by hour")
    save_plot(fig, output_dir / "hourly_richness")


def plot_species_circadian(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].plot(summary["hour"], summary["prop_species1_calling"], color="tab:purple")
    axes[0].set_title("Species 1 calling")
    axes[1].plot(summary["hour"], summary["prop_species2_calling"], color="tab:orange")
    axes[1].set_title("Species 2 calling")
    for ax in axes:
        ax.set_xlabel("Hour")
        ax.set_ylabel("Proportion calling")
    save_plot(fig, output_dir / "species_circadian_calling")


def plot_daily_effort(daily_effort: pd.DataFrame, output_dir: Path) -> None:
    long_df = daily_effort.melt(
        id_vars="day_id",
        value_vars=["species1_daily_calling_effort", "species2_daily_calling_effort"],
        var_name="species",
        value_name="daily_calling_effort",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=long_df, x="species", y="daily_calling_effort", ax=ax, inner=None)
    sns.boxplot(data=long_df, x="species", y="daily_calling_effort", ax=ax, width=0.25)
    ax.set_xlabel("Species")
    ax.set_ylabel("Daily calling effort (hours)")
    ax.set_title("Daily calling effort comparison")
    save_plot(fig, output_dir / "daily_calling_effort")


def _reference_value(series: pd.Series):
    if series.dropna().empty:
        return np.nan
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return series.dropna().iloc[0]


def make_effect_plot(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    df: pd.DataFrame,
    predictor: str,
    output_path: Path,
    day_id_col: str = "day_id",
    group_col: Optional[str] = None,
    group_values: Optional[List[int]] = None,
) -> None:
    base = {}
    orig_exog = result.model.data.orig_exog
    if predictor not in orig_exog.columns:
        return
    if "hour" in orig_exog.columns:
        base["hour"] = _reference_value(df["hour"])
    base[day_id_col] = _reference_value(df[day_id_col])

    if group_col and group_col not in orig_exog.columns:
        group_col = None
        group_values = None

    for col in orig_exog.columns:
        if col in base or col == predictor or col == day_id_col or col == group_col:
            continue
        base[col] = df[col].mean()

    if predictor == "hour":
        grid = np.arange(0, 24)
    else:
        values = df[predictor].dropna()
        grid = np.linspace(values.min(), values.max(), 50) if not values.empty else np.array([])

    plot_rows = []
    if group_col and group_values:
        for gv in group_values:
            for val in grid:
                row = base.copy()
                row[predictor] = val
                row[group_col] = gv
                plot_rows.append(row)
    else:
        for val in grid:
            row = base.copy()
            row[predictor] = val
            plot_rows.append(row)

    if not plot_rows:
        return
    plot_df = pd.DataFrame(plot_rows)
    preds = result.predict(plot_df)
    plot_df["prediction"] = preds

    fig, ax = plt.subplots(figsize=(6, 4))
    if group_col and group_values:
        for gv in group_values:
            subset = plot_df[plot_df[group_col] == gv]
            ax.plot(subset[predictor], subset["prediction"], label=f"{group_col}={gv}")
        ax.legend(title=group_col)
    else:
        ax.plot(plot_df[predictor], plot_df["prediction"], color="tab:blue")

    ax.set_xlabel(predictor.replace("_", " ").title())
    ax.set_ylabel("Fitted mean")
    ax.set_title(f"Effect of {predictor}")
    save_plot(fig, output_path)


def wilcoxon_effort_test(daily_effort: pd.DataFrame) -> Dict[str, float]:
    paired = daily_effort[["species1_daily_calling_effort", "species2_daily_calling_effort"]].dropna()
    s1 = paired["species1_daily_calling_effort"]
    s2 = paired["species2_daily_calling_effort"]
    if len(s1) == 0:
        return {"statistic": np.nan, "p_value": np.nan, "n": 0}
    try:
        stat, p = stats.wilcoxon(s1, s2)
        return {"statistic": stat, "p_value": p, "n": len(s1)}
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(s1)}


def summarize_effort(daily_effort: pd.DataFrame) -> pd.DataFrame:
    paired = daily_effort[["species1_daily_calling_effort", "species2_daily_calling_effort"]].dropna()

    def stats_summary(series: pd.Series) -> Dict[str, float]:
        return {
            "n": series.count(),
            "mean": series.mean(),
            "sd": series.std(),
            "median": series.median(),
            "iqr": series.quantile(0.75) - series.quantile(0.25),
        }

    return pd.DataFrame(
        {
            "species1": stats_summary(paired["species1_daily_calling_effort"]),
            "species2": stats_summary(paired["species2_daily_calling_effort"]),
        }
    )


def select_env_predictor(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    config: AnalysisConfig,
) -> Optional[str]:
    formula = result.model.formula
    if config.humidity_col in formula:
        return config.humidity_col
    if config.temp_col in formula:
        return config.temp_col
    return None


def write_summary_report(
    output_path: Path,
    config: AnalysisConfig,
    qa_summary: pd.DataFrame,
    correlation: Dict[str, float],
    day_night_summary: pd.DataFrame,
    effort_summary: pd.DataFrame,
    wilcoxon_result: Dict[str, float],
    model_choices: Dict[str, str],
    modeling_notes: List[str],
) -> None:
    def table_block(df: pd.DataFrame, index: bool = True) -> str:
        return "```\n" + df.to_string(index=index) + "\n```"

    lines = [
        "# Calling analysis summary",
        "",
        "## Data quality",
        table_block(qa_summary, index=False),
        "",
        f"Temperature-humidity correlation (n={correlation['n']}): r={correlation['r']:.3f}, p={correlation['p_value']:.3g}",
        "",
        "## Day vs night calling (means)",
        table_block(day_night_summary),
        "",
        "## Daily calling effort summary",
        table_block(effort_summary),
        "",
        "### Wilcoxon signed-rank test",
        f"n={wilcoxon_result['n']}, statistic={wilcoxon_result['statistic']}, p={wilcoxon_result['p_value']}",
        "",
        "## Results summary",
        f"- Richness model selected by AIC: {model_choices.get('richness', 'n/a')}",
        f"- Any-calling model selected by AIC: {model_choices.get('any_calling', 'n/a')}",
        f"- Species 1 presence model selected by AIC: {model_choices.get('species1_presence', 'n/a')}",
        f"- Species 2 presence model selected by AIC: {model_choices.get('species2_presence', 'n/a')}",
        f"- Daily effort (paired days): species1 mean={effort_summary.loc['mean', 'species1']:.2f}, "
        f"species2 mean={effort_summary.loc['mean', 'species2']:.2f}",
        "",
        "## Selected models (AIC)",
    ]
    for model_name, choice in model_choices.items():
        lines.append(f"- {model_name}: {choice}")
    lines.extend(["", "## Modeling approach notes"])
    lines.extend([f"- {note}" for note in modeling_notes])
    output_path.write_text("\n".join(lines))


def prepare_model_dataset(
    df: pd.DataFrame,
    response: str,
    predictors: List[str],
    config: AnalysisConfig,
) -> pd.DataFrame:
    cols = list(dict.fromkeys([response, "day_id"] + predictors))
    return df[cols].dropna().copy()


def run_models_for_response(
    df: pd.DataFrame,
    response: str,
    family: sm.families.Family,
    predictor_sets: List[Tuple[str, List[str]]],
    config: AnalysisConfig,
    output_tables: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, sm.regression.linear_model.RegressionResultsWrapper]:
    predictor_union: List[str] = []
    for _, predictors in predictor_sets:
        predictor_union.extend(predictors)
    predictor_union = list(dict.fromkeys(predictor_union))
    df_model = prepare_model_dataset(df, response, predictor_union, config)
    results = fit_candidate_models(
        df_model,
        response,
        predictor_sets,
        family,
        hour_as_categorical=config.hour_as_categorical,
    )
    selection = model_selection_table(results)
    save_table(selection, output_tables / f"{response}_model_selection.csv")

    best_name = selection.loc[0, "model"]
    best_result = results[best_name]
    coefs = coefficient_table(best_result)
    save_table(coefs, output_tables / f"{response}_coefficients.csv")
    return selection, coefs, best_name, best_result


def main(config: AnalysisConfig) -> None:
    sns.set_theme(style="whitegrid")
    output_dirs = ensure_output_dirs(config.output_dir)

    df = load_data(config.input_path, config.datetime_col)
    df = prepare_data(df, config)

    qa = data_quality_checks(df, config)
    save_table(qa["missing"].reset_index().rename(columns={"index": "column"}), output_dirs["tables"] / "missing_values.csv")
    save_table(qa["obs_by_day"].reset_index(), output_dirs["tables"] / "observations_by_day.csv")
    save_table(qa["summary"], output_dirs["tables"] / "quality_summary.csv")

    complete_df = get_complete_days(df)
    complete_df.to_csv(output_dirs["data"] / "complete_days_hourly.csv", index=False)
    df.to_csv(output_dirs["data"] / "cleaned_hourly.csv", index=False)

    daily_effort = compute_daily_effort(complete_df)
    daily_effort.to_csv(output_dirs["data"] / "daily_calling_effort.csv", index=False)

    correlation = compute_temp_humidity_correlation(df, config)
    circadian = summarize_circadian(complete_df, config)
    circadian.to_csv(output_dirs["tables"] / "circadian_summary.csv", index=False)
    day_night = summarize_day_night(df)

    plot_environmental_circadian(circadian, output_dirs["plots"])
    plot_environmental_timeseries(df, config, output_dirs["plots"])
    plot_hourly_richness(circadian, output_dirs["plots"])
    plot_species_circadian(circadian, output_dirs["plots"])
    plot_daily_effort(daily_effort, output_dirs["plots"])

    predictors_base = [
        ("intercept_only", []),
        ("hour", ["hour"]),
        ("temperature", [config.temp_col]),
        ("humidity", [config.humidity_col]),
        ("hour_temperature", ["hour", config.temp_col]),
        ("hour_humidity", ["hour", config.humidity_col]),
    ]

    model_choices = {}
    modeling_notes = [
        "Models are fit as GLMs with day fixed effects to account for repeated hourly observations within day.",
        "GLM with day fixed effects was chosen to enable AIC-based model selection; GEE does not provide AIC.",
    ]
    if not np.isnan(correlation["r"]):
        if abs(correlation["r"]) >= config.collinearity_threshold:
            modeling_notes.append(
                "Temperature and humidity are strongly correlated; they are not included together in any model."
            )
        else:
            modeling_notes.append(
                "Temperature and humidity are not strongly correlated; models still evaluate them separately per paper."
            )
    modeling_notes.append(
        "Model selection uses complete cases for all candidate predictors, which may exclude hours with missing humidity."
    )

    richness_sel, richness_coef, richness_best, richness_best_result = run_models_for_response(
        df,
        "richness",
        sm.families.Poisson(),
        predictors_base,
        config,
        output_dirs["tables"],
    )
    model_choices["richness"] = richness_best

    any_sel, any_coef, any_best, any_best_result = run_models_for_response(
        df,
        "any_calling",
        sm.families.Binomial(),
        predictors_base,
        config,
        output_dirs["tables"],
    )
    model_choices["any_calling"] = any_best

    species1_predictors = predictors_base + [
        ("hour_temperature_other", ["hour", config.temp_col, "other_species_calling_for_species1"]),
        ("hour_humidity_other", ["hour", config.humidity_col, "other_species_calling_for_species1"]),
    ]

    species1_sel, species1_coef, species1_best, species1_best_result = run_models_for_response(
        df,
        "species1_presence",
        sm.families.Binomial(),
        species1_predictors,
        config,
        output_dirs["tables"],
    )
    model_choices["species1_presence"] = species1_best
    species1_env = select_env_predictor(species1_best_result, config)

    species2_predictors = predictors_base + [
        ("hour_temperature_other", ["hour", config.temp_col, "other_species_calling_for_species2"]),
        ("hour_humidity_other", ["hour", config.humidity_col, "other_species_calling_for_species2"]),
    ]

    species2_sel, species2_coef, species2_best, species2_best_result = run_models_for_response(
        df,
        "species2_presence",
        sm.families.Binomial(),
        species2_predictors,
        config,
        output_dirs["tables"],
    )
    model_choices["species2_presence"] = species2_best
    species2_env = select_env_predictor(species2_best_result, config)

    sensitivity_predictor_list = ["hour", "other_species_calling_for_species1", "species1_abundance"]
    if species1_env:
        sensitivity_predictor_list.append(species1_env)
    sensitivity_predictors = [("sensitivity_abundance", sensitivity_predictor_list)]
    df_sensitivity = prepare_model_dataset(df, "species1_presence", sensitivity_predictor_list, config)
    sensitivity_results = fit_candidate_models(
        df_sensitivity,
        "species1_presence",
        sensitivity_predictors,
        sm.families.Binomial(),
        hour_as_categorical=config.hour_as_categorical,
    )
    sensitivity_table = model_selection_table(sensitivity_results)
    save_table(sensitivity_table, output_dirs["tables"] / "species1_presence_sensitivity_model_selection.csv")
    sensitivity_coeffs = coefficient_table(sensitivity_results[sensitivity_table.loc[0, "model"]])
    save_table(sensitivity_coeffs, output_dirs["tables"] / "species1_presence_sensitivity_coefficients.csv")
    modeling_notes.append(
        "Sensitivity model for species 1 includes abundance; interpret cautiously because abundance derives from presence."
    )

    effect_dir = output_dirs["plots"]
    richness_env = select_env_predictor(richness_best_result, config)
    make_effect_plot(richness_best_result, df, "hour", effect_dir / "richness_effect_hour")
    if richness_env:
        make_effect_plot(richness_best_result, df, richness_env, effect_dir / f"richness_effect_{richness_env}")

    any_env = select_env_predictor(any_best_result, config)
    make_effect_plot(any_best_result, df, "hour", effect_dir / "any_calling_effect_hour")
    if any_env:
        make_effect_plot(any_best_result, df, any_env, effect_dir / f"any_calling_effect_{any_env}")

    make_effect_plot(
        species1_best_result,
        df,
        "hour",
        effect_dir / "species1_effect_hour",
        group_col="other_species_calling_for_species1",
        group_values=[0, 1],
    )
    if species1_env:
        make_effect_plot(
            species1_best_result,
            df,
            species1_env,
            effect_dir / f"species1_effect_{species1_env}",
            group_col="other_species_calling_for_species1",
            group_values=[0, 1],
        )

    make_effect_plot(
        species2_best_result,
        df,
        "hour",
        effect_dir / "species2_effect_hour",
        group_col="other_species_calling_for_species2",
        group_values=[0, 1],
    )
    if species2_env:
        make_effect_plot(
            species2_best_result,
            df,
            species2_env,
            effect_dir / f"species2_effect_{species2_env}",
            group_col="other_species_calling_for_species2",
            group_values=[0, 1],
        )

    effort_summary = summarize_effort(daily_effort)
    save_table(effort_summary.reset_index().rename(columns={"index": "metric"}), output_dirs["tables"] / "daily_effort_summary.csv")
    wilcoxon_result = wilcoxon_effort_test(daily_effort)

    write_summary_report(
        output_dirs["report"] / "summary.md",
        config,
        qa["summary"],
        correlation,
        day_night,
        effort_summary,
        wilcoxon_result,
        model_choices,
        modeling_notes,
    )


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Run calling analysis pipeline.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", default="calling_analysis/outputs", help="Output directory.")
    parser.add_argument("--datetime-col", default="DateTime", help="Datetime column name.")
    parser.add_argument("--temp-col", default="Temp", help="Temperature column name.")
    parser.add_argument("--humidity-col", default="RH%", help="Humidity column name.")
    parser.add_argument("--species1-col", default="Gastrotheca chrysosticta", help="Species 1 call index column.")
    parser.add_argument("--species2-col", default="Oreobates berdemenos", help="Species 2 call index column.")
    parser.add_argument("--rain-col", default="Rain", help="Optional rain column.")
    parser.add_argument(
        "--hour-as-numeric",
        action="store_false",
        dest="hour_as_categorical",
        help="Treat hour as numeric predictor.",
    )
    parser.add_argument("--collinearity-threshold", type=float, default=0.7, help="Correlation threshold for collinearity.")
    args = parser.parse_args()
    return AnalysisConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        datetime_col=args.datetime_col,
        temp_col=args.temp_col,
        humidity_col=args.humidity_col,
        species1_col=args.species1_col,
        species2_col=args.species2_col,
        rain_col=args.rain_col,
        hour_as_categorical=args.hour_as_categorical,
        collinearity_threshold=args.collinearity_threshold,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = parse_args()
    main(cfg)
