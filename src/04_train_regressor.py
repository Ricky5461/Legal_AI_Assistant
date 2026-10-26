import h2o
from h2o.automl import H2OAutoML
from pathlib import Path

DATA_PATH = (
    "data/raw/"
    "Model2abbaselineData_2010to2018_merged_criminal_regr.csv.gz"
)

# REPLACE with the actual duration column after inspecting the schema.
TARGET = "actual_duration_months"

Path("models/regression").mkdir(
    parents=True,
    exist_ok=True
)

h2o.init(max_mem_size="6G", nthreads=4)

try:
    data = h2o.import_file(DATA_PATH)

    print("Columns:", data.columns)
    print("Rows:", data.nrows)

    if TARGET not in data.columns:
        raise ValueError(
            f"Target '{TARGET}' not found. "
            "Inspect the regression dataset before training."
        )

    data[TARGET] = data[TARGET].asnumeric()

    data = data[data[TARGET].isna() == 0]

    # Pilot only: keep the sample manageable.
    data = data.head(100_000)

    train, valid, test = data.split_frame(
        ratios=[0.70, 0.15],
        seed=42
    )

    excluded = {
        TARGET,
        "case_id",
        "filing_date",
        "disposal_date",
        "pendency_model_m"
    }

    features = [
        c for c in train.columns
        if c not in excluded
    ]

    automl = H2OAutoML(
        max_models=5,
        max_runtime_secs=600,
        seed=42,
        sort_metric="RMSE",
        exclude_algos=["XGBoost", "DeepLearning"]
    )

    automl.train(
        x=features,
        y=TARGET,
        training_frame=train,
        validation_frame=valid
    )

    print(automl.leaderboard)
    print(automl.leader.model_performance(test))

    print(
        h2o.save_model(
            model=automl.leader,
            path="models/regression"
        )
    )

finally:
    h2o.shutdown(prompt=False)