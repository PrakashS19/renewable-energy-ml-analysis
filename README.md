# Renewable Energy Machine Learning Analysis

This repository supports **RMIT Case Studies in Data Science – Individual Task 1**. It reproduces the machine-learning analysis used in the executive summary.

## Case-study purpose

The analysis evaluates whether machine-learning models can predict renewable-energy power output from environmental, operational and temporal variables. The case study is connected to the selected **Alinta Energy Data Scientist** role through predictive modelling, complex-data analysis, evidence-based decision-making and the organisation's renewable-energy context.

## Public datasets

1. **Solar Power Generation Data – Kaggle**  
   https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

2. **Wind Turbine SCADA Dataset – Kaggle**  
   https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

The raw datasets are not redistributed in this repository. Download and extract them into the folder structure shown below.

## Required folder structure

```text
renewable-energy-ml-analysis/
├── analysis.py
├── requirements.txt
├── README.md
├── solar_datasets/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
├── wind_datasets/
│   └── T1.csv
└── results/
```

## Installation

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run the analysis

```bash
python analysis.py
```

The script validates the input files, preprocesses both datasets, trains **Linear Regression** and **LightGBM**, evaluates the models and writes reproducible evidence to `results/`.

## Evaluation design

The analysis uses a **chronological 80/20 train-test split** rather than random shuffling because both datasets contain time-dependent observations. This better reflects the intended use case: learning from earlier observations and evaluating on later unseen data.

Three regression metrics are reported:

- **MAE**: average absolute prediction error in the target's original units; easy to interpret operationally.
- **RMSE**: penalises larger errors more strongly, useful because large generation-prediction errors may be more consequential.
- **R²**: measures the proportion of observed target variation explained by the model and supports comparison of overall fit.

No single metric is treated as sufficient on its own.

## Feature decisions

### Solar

Target: `AC_POWER`

Predictors include:

- ambient temperature,
- module temperature,
- irradiation,
- plant identifier,
- cyclic hour features,
- cyclic day-of-year features.

`DC_POWER`, `DAILY_YIELD` and `TOTAL_YIELD` are deliberately excluded from the predictor set. They are generation-related variables and would make prediction of contemporaneous `AC_POWER` less representative of forecasting from environmental and temporal conditions.

### Wind

Target: `LV ActivePower (kW)`

Predictors include:

- wind speed,
- theoretical manufacturer power curve,
- wind direction encoded as sine/cosine,
- cyclic hour features,
- cyclic day-of-year features.

Wind direction is circular, so sine/cosine encoding avoids treating 359° and 1° as numerically far apart.

## Reproduced results

Expected results from the supplied Kaggle files are approximately:

| Dataset | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Solar | Linear Regression | 1560.783 | 2156.624 | 0.9011 |
| Solar | LightGBM | 453.321 | 1327.760 | 0.9625 |
| Wind | Linear Regression | 247.002 | 409.494 | 0.9069 |
| Wind | LightGBM | 220.599 | 436.211 | 0.8944 |

### Interpretation

For the solar data, LightGBM performs substantially better across all three metrics, indicating important nonlinear relationships between irradiation, temperature, time and power output.

For the wind data, the comparison is mixed. LightGBM achieves the lower MAE, while Linear Regression achieves the lower RMSE and higher R². This means the two datasets provide **complementary rather than identical evidence**: model complexity is highly valuable for the solar case, but the simpler model generalises competitively for the wind case.

## Outputs

A successful run creates:

```text
results/
├── dataset_summary.csv
├── model_metrics.csv
├── test_predictions.csv
├── run_summary.txt
├── solar_rmse_comparison.png
├── solar_actual_vs_predicted.png
├── wind_rmse_comparison.png
└── wind_actual_vs_predicted.png
```

These files provide process evidence for the report appendix and make the reported results independently reproducible.

## Limitations

- The solar dataset covers only about 34 days and two plants in India, limiting seasonal and geographic generalisation.
- The wind dataset represents one turbine/site in Turkey, so conclusions should not be assumed to transfer directly to Australian assets.
- The datasets do not contain Alinta Energy customer, pricing or commercial data. Their relevance is through predictive modelling and renewable-energy analytics rather than representing the exact day-to-day data of the advertised role.
- The analysis is predictive/diagnostic and does not establish causal relationships.
- No production deployment, probabilistic forecasting or external weather forecast data are included.

## Reproducibility

LightGBM uses `random_state=42`. Running `python analysis.py` on the specified source files should reproduce the reported metrics to normal floating-point/package-version tolerance.
