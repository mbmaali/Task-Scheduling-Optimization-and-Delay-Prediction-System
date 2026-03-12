# Workflow Task Scheduling Optimization and Delay Prediction System

An Industrial Engineering portfolio project that analyzes workflow operations data, estimates delay risk, and optimizes employee-task assignments using a simple linear programming model.

This repository is organized as a clean GitHub-ready version of the project. The strongest part of the work is the workflow analysis and scheduling optimization. The delay prediction model is included as an exploratory component and is documented honestly below.

## Repository Note

The raw dataset is **not included** in this repository.

Dataset source:
- Kaggle: `https://www.kaggle.com/datasets/algozee/workflow-operations-performance-dataset/data`

To run the project, download the CSV and place it at:
- `data/workflow_data.csv`

Instructions are also included in [data/README.md](data/README.md).

## Project Focus

This project demonstrates:
- workflow operations analysis
- delay-risk modeling
- task assignment optimization
- workload balancing
- cost comparison
- clear reporting for an internship portfolio

## What The Project Does

1. Cleans and validates workflow task data.
2. Creates timing, delay, efficiency, workload, and cost features.
3. Produces exploratory analysis and KPI charts.
4. Trains several classification models to estimate whether a task may be delayed.
5. Uses predicted delay risk inside a PuLP optimization model to reassign tasks to employees.
6. Compares original vs optimized assignments using workload, delay-risk, and cost metrics.

## Technical Honesty

This project is publishable, but the prediction model should be described carefully.

Important points:
- The delay model does **not** use obvious post-completion leakage features such as `Actual_Time_Minutes`, `Delay_Flag`, `Calculated_Delay_Minutes`, `Task_End_Time`, or `Task_Duration_From_Timestamps` as inputs.
- However, the dataset is **extremely imbalanced**: most tasks are already labeled as delayed.
- Because of that imbalance, the current Random Forest benchmark looks strong on positive-class F1, but in practice it mostly predicts the majority class.
- The optimization and operations-analysis portions of the project are more informative and more impressive than the current classifier performance.

In other words: this is a strong portfolio project for analytics and optimization, but the delay predictor should be presented as an exploratory benchmark, not as a production-ready forecasting model.

## Dataset Columns

Expected input columns:
- `Workflow_ID`
- `Process_Name`
- `Task_ID`
- `Task_Type`
- `Priority_Level`
- `Department`
- `Assigned_Employee_ID`
- `Task_Start_Time`
- `Task_End_Time`
- `Estimated_Time_Minutes`
- `Actual_Time_Minutes`
- `Delay_Flag`
- `Approval_Level`
- `Employee_Workload`
- `Cost_Per_Task`

## Derived Features

The code creates additional fields such as:
- `Calculated_Delay_Minutes`
- `Is_Delayed_Binary`
- `Task_Duration_From_Timestamps`
- `Start_Hour`
- `Start_DayOfWeek`
- `Task_Efficiency`
- `Cost_Per_Minute`

Only a subset of pre-task or planning-time features is used for the prediction model.

## Methods

### Data Preparation
- standardize column names
- handle missing values
- normalize delay flags
- convert timestamps
- remove duplicates
- create engineered timing and cost features

### Exploratory Analysis
The analysis includes:
- delay rate overall
- delay rate by department, priority, and task type
- average actual vs estimated time
- workload analysis
- cost by department
- workload vs delay relationship
- duration distribution
- correlation heatmap
- IE-style KPIs

### Delay Prediction
Models included:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost if installed

Prediction features:
- `Task_Type`
- `Priority_Level`
- `Department`
- `Approval_Level`
- `Employee_Workload`
- `Estimated_Time_Minutes`
- `Cost_Per_Task`
- `Start_Hour`
- `Start_DayOfWeek`

### Optimization Model
The optimization model assigns a sample of tasks to candidate employees by minimizing a weighted combination of:
- predicted delay risk
- workload imbalance
- estimated cost

Constraints include:
- each task assigned exactly once
- employee planning capacity limits
- optional maximum tasks per employee
- optional department compatibility logic

This is the main Industrial Engineering contribution of the repository.

## Sample Results From The Current Dataset

These files are included as sample outputs from a verified local run:
- [outputs/reports/model_comparison_metrics.csv](outputs/reports/model_comparison_metrics.csv)
- [outputs/reports/before_vs_after_metrics.csv](outputs/reports/before_vs_after_metrics.csv)
- [outputs/reports/eda_kpis.csv](outputs/reports/eda_kpis.csv)
- [outputs/optimized_assignments.csv](outputs/optimized_assignments.csv)

Optimization improvements from the reference run:
- total estimated cost: `$4400.38` to `$3953.12`
- workload standard deviation: `186.69` to `127.48`
- overloaded employees: `6` to `3`
- total predicted delay risk: `24.87` to `24.51`

Prediction summary from the same run:
- best selected model: `Random Forest`
- reported F1 score: `0.9744`
- reported recall: `1.0000`
- reported ROC-AUC: `0.5576`

Interpretation:
- The optimization results are useful.
- The prediction metrics should be interpreted carefully because of class imbalance.

## Limitations

- The target is highly imbalanced, which makes standard accuracy and positive-class F1 look stronger than the model really is.
- The current best classifier behaves very similarly to a majority-class predictor on this dataset.
- `Start_Hour` and `Start_DayOfWeek` are derived from actual start timestamps, so they are only appropriate if start timing is known before scoring.
- Employee capacity is estimated from historical workload proxies because exact staffing capacity is not available.
- The optimization model is a what-if planning tool, not a production scheduling engine.

## Project Structure

```text
workflow-optimization-project/
├── data/
│   └── README.md
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── exploratory_analysis.py
│   ├── delay_prediction.py
│   ├── optimization_model.py
│   ├── evaluation.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── optimized_assignments.csv
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How To Run

### 1. Create a virtual environment

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the Kaggle CSV at:
```text
data/workflow_data.csv
```

### 4. Run the pipeline

```bash
python main.py
```

### 5. Open the notebook

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```



- use a time-based validation split instead of a purely random split
- add better class-imbalance handling and threshold tuning
- introduce employee skill data and stronger compatibility constraints
- use scheduled start information instead of actual start timestamps
- add scenario testing for different objective weights in the optimization model
