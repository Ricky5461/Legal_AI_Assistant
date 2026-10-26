from pathlib import Path
import h2o
from h2o.automl import H2OAutoML

TARGET = "pendency_model_m"

Path("models/classification").mkdir(
    parents=True,
    exist_ok=True
)

h2o.init(
    max_mem_size="6G",
    nthreads=4
)

try:
    train = h2o.import_file("data/processed/train.csv")
    valid = h2o.import_file("data/processed/valid.csv")
    test = h2o.import_file("data/processed/test.csv")

    for frame in (train, valid, test):
        frame[TARGET] = frame[TARGET].asfactor()

    excluded = {
        TARGET,
        "case_id",
        "filing_date",
        "disposal_date",
        "actual_duration_months"
    }

    features = [
        c for c in train.columns
        if c not in excluded
    ]

    automl = H2OAutoML(
        max_models=5,
        max_runtime_secs=600,
        seed=42,
        sort_metric="AUC",
        exclude_algos=["XGBoost", "DeepLearning"]
    )

    automl.train(
        x=features,
        y=TARGET,
        training_frame=train,
        validation_frame=valid
    )

    print("\nAutoML leaderboard:")
    print(automl.leaderboard)

    print("\nTest performance:")
    print(automl.leader.model_performance(test))

    model_path = h2o.save_model(
        model=automl.leader,
        path="models/classification"
    )

    print("\nSaved model:", model_path)

finally:
    h2o.shutdown(prompt=False)