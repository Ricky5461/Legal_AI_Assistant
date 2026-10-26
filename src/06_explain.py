import h2o

MODEL_PATH = "models/classification"
TEST_PATH = "data/processed/test.csv"

h2o.init(max_mem_size="4G", nthreads=4)

try:
    model = h2o.load_model(MODEL_PATH)
    test = h2o.import_file(TEST_PATH)

    print("Variable importance:")
    print(model.varimp(use_pandas=True))

    print("\nModel performance:")
    print(model.model_performance(test))

    # Depending on the model type and H2O version,
    # not every explanation method will be supported.
    print("\nBuilt-in model summary:")
    print(model.summary())

finally:
    h2o.shutdown(prompt=False)