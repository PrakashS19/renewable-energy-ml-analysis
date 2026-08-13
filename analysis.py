"""
Renewable Energy Machine Learning Analysis
RMIT Case Studies in Data Science - Individual Task 1

Purpose
-------
Compare Linear Regression and LightGBM on:
1. Solar photovoltaic generation data from two plants in India.
2. Wind turbine SCADA data from Turkey.

The script:
- validates input files,
- reports dataset dimensions and missing values,
- engineers cyclic time/direction features,
- applies a chronological 80/20 train-test split,
- trains both models,
- evaluates MAE, RMSE and R²,
- exports reproducible tables and figures.

Raw Kaggle datasets are not included in this repository.
Place them in:
    solar_datasets/
    wind_datasets/
next to this script before running.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

BASE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE,
        help="Folder containing solar_datasets/ and wind_datasets/."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE / "results",
        help="Folder where tables and figures will be written."
    )
    return parser.parse_args()


def validate_inputs(solar_dir, wind_dir):
    required = [
        solar_dir / "Plant_1_Generation_Data.csv",
        solar_dir / "Plant_1_Weather_Sensor_Data.csv",
        solar_dir / "Plant_2_Generation_Data.csv",
        solar_dir / "Plant_2_Weather_Sensor_Data.csv",
        wind_dir / "T1.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required dataset file(s) not found:\n- " + "\n- ".join(missing)
        )


def add_time_features(df, dt_col):
    out = df.copy()
    dt = out[dt_col]
    hour = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    return out


def raw_dataset_summary(solar_dir, wind_dir):
    files = [
        ("Solar", "Plant 1 Generation", solar_dir / "Plant_1_Generation_Data.csv"),
        ("Solar", "Plant 1 Weather", solar_dir / "Plant_1_Weather_Sensor_Data.csv"),
        ("Solar", "Plant 2 Generation", solar_dir / "Plant_2_Generation_Data.csv"),
        ("Solar", "Plant 2 Weather", solar_dir / "Plant_2_Weather_Sensor_Data.csv"),
        ("Wind", "Turbine SCADA", wind_dir / "T1.csv"),
    ]
    rows = []
    for domain, name, path in files:
        df = pd.read_csv(path)
        rows.append({
            "domain": domain,
            "file": name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "attributes": ", ".join(df.columns.astype(str)),
        })
    return pd.DataFrame(rows)


def load_solar_plant(solar_dir, plant_no):
    generation = pd.read_csv(
        solar_dir / f"Plant_{plant_no}_Generation_Data.csv"
    )
    weather = pd.read_csv(
        solar_dir / f"Plant_{plant_no}_Weather_Sensor_Data.csv"
    )

    # The two generation files use different timestamp formats in the
    # published dataset.
    if plant_no == 1:
        generation["DATE_TIME"] = pd.to_datetime(
            generation["DATE_TIME"], format="%d-%m-%Y %H:%M"
        )
    else:
        generation["DATE_TIME"] = pd.to_datetime(
            generation["DATE_TIME"], format="%Y-%m-%d %H:%M:%S"
        )

    weather["DATE_TIME"] = pd.to_datetime(
        weather["DATE_TIME"], format="%Y-%m-%d %H:%M:%S"
    )

    # Generation data are inverter-level. Aggregate AC power to the plant
    # timestamp before joining with plant-level weather measurements.
    generation_agg = generation.groupby(
        ["DATE_TIME", "PLANT_ID"], as_index=False
    ).agg(AC_POWER=("AC_POWER", "sum"))

    weather = weather.drop(columns=["SOURCE_KEY"])
    merged = generation_agg.merge(
        weather, on=["DATE_TIME", "PLANT_ID"], how="inner"
    )
    merged["plant_num"] = plant_no
    return merged


def chronological_split(df, dt_col, fraction=0.8):
    unique_times = np.sort(df[dt_col].unique())
    split_index = int(len(unique_times) * fraction)
    cutoff = unique_times[split_index]

    train = df[df[dt_col] < cutoff].copy()
    test = df[df[dt_col] >= cutoff].copy()
    return train, test, cutoff


def model_definitions():
    return {
        "Linear Regression": LinearRegression(),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
    }


def evaluate_models(train, test, features, target, domain):
    metric_rows = []
    prediction_frames = []

    for model_name, model in model_definitions().items():
        model.fit(train[features], train[target])
        predictions = model.predict(test[features])

        metric_rows.append({
            "dataset": domain,
            "model": model_name,
            "MAE": mean_absolute_error(test[target], predictions),
            "RMSE": mean_squared_error(test[target], predictions) ** 0.5,
            "R2": r2_score(test[target], predictions),
            "train_rows": len(train),
            "test_rows": len(test),
        })

        prediction_frames.append(pd.DataFrame({
            "dataset": domain,
            "model": model_name,
            "actual": test[target].to_numpy(),
            "predicted": predictions,
        }))

    return pd.DataFrame(metric_rows), pd.concat(
        prediction_frames, ignore_index=True
    )


def save_metric_figure(metrics, dataset_name, results_dir):
    subset = metrics[metrics["dataset"] == dataset_name].copy()
    x = np.arange(len(subset))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, subset["RMSE"])
    ax.set_xticks(x)
    ax.set_xticklabels(subset["model"])
    ax.set_ylabel("RMSE")
    ax.set_title(f"{dataset_name}: RMSE by model")
    fig.tight_layout()
    fig.savefig(
        results_dir / f"{dataset_name.lower()}_rmse_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_actual_vs_predicted(predictions, dataset_name, results_dir):
    subset = predictions[predictions["dataset"] == dataset_name].copy()

    # Plot a deterministic sample when there are many observations to keep
    # the figure readable and the file compact.
    if len(subset) > 8000:
        subset = subset.sample(8000, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in subset["model"].unique():
        part = subset[subset["model"] == model_name]
        ax.scatter(
            part["actual"],
            part["predicted"],
            s=8,
            alpha=0.25,
            label=model_name,
        )

    low = min(subset["actual"].min(), subset["predicted"].min())
    high = max(subset["actual"].max(), subset["predicted"].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)

    ax.set_xlabel("Actual power")
    ax.set_ylabel("Predicted power")
    ax.set_title(f"{dataset_name}: actual vs predicted power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        results_dir / f"{dataset_name.lower()}_actual_vs_predicted.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    solar_dir = data_root / "solar_datasets"
    wind_dir = data_root / "wind_datasets"
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    validate_inputs(solar_dir, wind_dir)

    # ---------- Raw dataset evidence ----------
    summary = raw_dataset_summary(solar_dir, wind_dir)
    summary.to_csv(results_dir / "dataset_summary.csv", index=False)

    # ---------- Solar ----------
    solar = pd.concat(
        [
            load_solar_plant(solar_dir, 1),
            load_solar_plant(solar_dir, 2),
        ],
        ignore_index=True,
    )
    solar = add_time_features(
        solar.sort_values("DATE_TIME").reset_index(drop=True),
        "DATE_TIME",
    )

    # DC_POWER, DAILY_YIELD and TOTAL_YIELD are deliberately excluded.
    # They are generation-related variables that would make AC_POWER
    # prediction less representative of forecasting from environmental
    # and temporal conditions.
    solar_features = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "plant_num",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    solar_train, solar_test, solar_cutoff = chronological_split(
        solar, "DATE_TIME", fraction=0.8
    )
    solar_metrics, solar_predictions = evaluate_models(
        solar_train,
        solar_test,
        solar_features,
        "AC_POWER",
        "Solar",
    )

    # ---------- Wind ----------
    wind = pd.read_csv(wind_dir / "T1.csv")
    wind["Date/Time"] = pd.to_datetime(
        wind["Date/Time"], format="%d %m %Y %H:%M"
    )
    wind = add_time_features(
        wind.sort_values("Date/Time").reset_index(drop=True),
        "Date/Time",
    )

    # Wind direction is circular: e.g. 359° and 1° are close directions.
    angle = np.deg2rad(wind["Wind Direction (°)"])
    wind["dir_sin"] = np.sin(angle)
    wind["dir_cos"] = np.cos(angle)

    wind_features = [
        "Wind Speed (m/s)",
        "Theoretical_Power_Curve (KWh)",
        "dir_sin",
        "dir_cos",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    wind_train, wind_test, wind_cutoff = chronological_split(
        wind, "Date/Time", fraction=0.8
    )
    wind_metrics, wind_predictions = evaluate_models(
        wind_train,
        wind_test,
        wind_features,
        "LV ActivePower (kW)",
        "Wind",
    )

    # ---------- Export evidence ----------
    all_metrics = pd.concat(
        [solar_metrics, wind_metrics], ignore_index=True
    )
    all_predictions = pd.concat(
        [solar_predictions, wind_predictions], ignore_index=True
    )

    all_metrics.to_csv(results_dir / "model_metrics.csv", index=False)
    all_predictions.to_csv(
        results_dir / "test_predictions.csv", index=False
    )

    save_metric_figure(all_metrics, "Solar", results_dir)
    save_metric_figure(all_metrics, "Wind", results_dir)
    save_actual_vs_predicted(
        all_predictions, "Solar", results_dir
    )
    save_actual_vs_predicted(
        all_predictions, "Wind", results_dir
    )

    summary_lines = [
        "RENEWABLE ENERGY ML ANALYSIS - REPRODUCIBLE RUN",
        "=" * 56,
        "",
        "DATASET SUMMARY",
        summary.to_string(index=False),
        "",
        f"Solar chronological cutoff: {solar_cutoff}",
        f"Solar train rows: {len(solar_train)}",
        f"Solar test rows: {len(solar_test)}",
        "",
        f"Wind chronological cutoff: {wind_cutoff}",
        f"Wind train rows: {len(wind_train)}",
        f"Wind test rows: {len(wind_test)}",
        "",
        "MODEL METRICS",
        all_metrics.to_string(index=False),
        "",
        "Method notes:",
        "- 80/20 split is chronological rather than randomly shuffled.",
        "- Solar target: AC_POWER.",
        "- Solar generation-related leakage variables were excluded.",
        "- Wind target: LV ActivePower (kW).",
        "- Wind direction was encoded with sine/cosine components.",
        "- random_state=42 is used for LightGBM reproducibility.",
    ]

    run_summary = "\n".join(summary_lines)
    (results_dir / "run_summary.txt").write_text(
        run_summary, encoding="utf-8"
    )

    print(run_summary)
    print(f"\nResults written to: {results_dir}")


if __name__ == "__main__":
    main()
