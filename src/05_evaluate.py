import pandas as pd
import h2o
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

MODEL_PATH = "models/classification"
TEST_PATH = "data/processed/test.csv"
TARGET = "pendency_model_m"

h2o.init(max_mem_size="4G", nthreads=4)

try:
    model = h2o.load_model(MODEL_PATH)
    test = h2o.import_file(TEST_PATH)

    predictions = model.predict(test).as_data_frame()
    actual = test[TARGET].as_data_frame().iloc[:, 0].astype(str)

    predicted = predictions["predict"].astype(str)

    print("\nClassification report:")
    print(classification_report(actual, predicted))

    print("\nConfusion matrix:")
    print(confusion_matrix(actual, predicted))

    if "delayed" in predictions.columns and actual.nunique() == 2:
        y_binary = (actual == "delayed").astype(int)
        print(
            "ROC-AUC:",
            roc_auc_score(y_binary, predictions["delayed"])
        )

finally:
    h2o.shutdown(prompt=False)