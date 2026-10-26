from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    "data/raw/Model1abData_2010to2018_merged_criminal_final.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "pendency_model_m"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# Read only a sample. Do not load the entire dataset.
df = pd.read_csv(
    DATA_PATH,
    nrows=SAMPLE_SIZE,
    low_memory=False
)

print("Sample shape:", df.shape)
print("Columns:", df.columns.tolist())

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        "Check the actual column names first."
    )

df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])

print("\nTarget distribution:")
print(df[TARGET].value_counts())

# Remove sensitive attributes from the initial predictive model.
sensitive_columns = [
    c for c in df.columns
    if any(term in c.lower() for term in [
        "muslim",
        "female_defendant",
        "female_petitioner",
        "female_adv",
        "female_judge"
    ])
]

# Exclude fields that may reveal information from after filing.
# Only remove columns that actually exist.
potential_leakage = [
    "decision_judge_tenure",
    "judgediff",
    "disposal_date",
    "actual_duration_months"
]

drop_columns = sensitive_columns + [
    c for c in potential_leakage if c in df.columns
]

df = df.drop(columns=drop_columns, errors="ignore")

# Keep the target as a clean categorical value.
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

valid_labels = {"delayed", "ontime"}
df = df[df[TARGET].isin(valid_labels)]

if df[TARGET].nunique() != 2:
    raise ValueError(
        "Expected both 'delayed' and 'ontime' labels. "
        "Check the target definition and values."
    )

# Split into training, validation, and test sets.
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df[TARGET]
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df[TARGET]
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
valid_df.to_csv(OUTPUT_DIR / "valid.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nData preparation completed.")
print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))
print("Test rows:", len(test_df))
print("Saved to:", OUTPUT_DIR)