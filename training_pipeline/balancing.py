import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

SEMI_TARGET_PROPORTIONS = {
    "severe_hypoglycemia": 0.05,
    "hypoglycemia": 0.10,
    "normoglycemia": 0.45,
    "hyperglycemia": 0.25,
    "severe_hyperglycemia": 0.15,
}

FULL_TARGET_PROPORTIONS = {
    "severe_hypoglycemia": 0.20,
    "hypoglycemia": 0.20,
    "normoglycemia": 0.20,
    "hyperglycemia": 0.20,
    "severe_hyperglycemia": 0.20,
}

# HELPERS
def get_glycemic_range(y):
    if y < 54:
        return "severe_hypoglycemia"
    elif y < 70:
        return "hypoglycemia"
    elif y < 181:
        return "normoglycemia"
    elif y < 251:
        return "hyperglycemia"
    else:
        return "severe_hyperglycemia"

def get_target_counts(total: int, target_proportions: dict) -> dict:
    target_counts = {
        range_name: int(total * proportion)
        for range_name, proportion in target_proportions.items()
    }

    missing = total - sum(target_counts.values())

    target_counts["normoglycemia"] += missing

    return target_counts

def _get_feature_columns(df: pd.DataFrame) -> list:
    x_cols = [
        col for col in df.columns
        if col.startswith("x") and col[1:].isdigit()
    ]

    return x_cols + ["y"]

# MAIN FUNCTION
def balance_training_set(df_train: pd.DataFrame, method: str, level: str) -> pd.DataFrame:

    if method not in ["random", "smoter", "smogn"]:
        raise ValueError("method must be: random, smoter or smogn")

    if level == "full":
        target_proportions = FULL_TARGET_PROPORTIONS
    elif level == "semi":
        target_proportions = SEMI_TARGET_PROPORTIONS
    else:
        raise ValueError("level must be: full or semi")

    if method == "random":
        return _random_balance(df_train, target_proportions)

    elif method == "smoter":
        return _smoter_balance(df_train, target_proportions)

    elif method == "smogn":
        print("SMOGN is not implemented yet.")

# DELEGATED FUNCTIONS
def _random_balance(df_train: pd.DataFrame, target_proportions: dict, random_state: int = 42) -> pd.DataFrame:

    # Add distribution summary before balancing

    df_train = df_train.copy()
    df_train["glycemic_range"] = df_train["y"].apply(get_glycemic_range)

    total_target = len(df_train)
    target_counts = get_target_counts(total_target, target_proportions)

    balanced_parts = []

    for range_name, target_n in target_counts.items():
        df_range = df_train[df_train["glycemic_range"] == range_name]

        current_n = len(df_range)

        if current_n == 0:
            print(f"{range_name} has 0 samples. Skipping.")
            continue

        if current_n > target_n:
            df_range_balanced = df_range.sample(n=target_n, replace=False, random_state=random_state)

        elif current_n < target_n:
            extra_n = target_n - current_n
            df_extra = df_range.sample(n=extra_n, replace=True, random_state=random_state)
            df_range_balanced = pd.concat([df_range, df_extra], ignore_index=True)

        else:
            df_range_balanced = df_range

        balanced_parts.append(df_range_balanced)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_balanced = df_balanced.drop(columns=["glycemic_range"])

    # Add distribution summary after balancing

    return df_balanced

def _smoter_balance(df_train: pd.DataFrame, target_proportions: dict, random_state: int = 42, k_neighbors: int = 5,) -> pd.DataFrame:

    # Add distribution summary before balancing

    df_train = df_train.copy()
    df_train["glycemic_range"] = df_train["y"].apply(get_glycemic_range)

    total_target = len(df_train)
    target_counts = get_target_counts(total_target, target_proportions)

    balanced_parts = []

    for range_name, target_n in target_counts.items():
        df_range = df_train[df_train["glycemic_range"] == range_name]

        current_n = len(df_range)

        if current_n == 0:
            print(f"{range_name} has 0 samples. Skipping.")
            continue

        if current_n > target_n:
            df_range_balanced = df_range.sample(n=target_n, replace=False, random_state=random_state)

        elif current_n < target_n:
            extra_n = target_n - current_n
            df_synthetic = _generate_smoter_samples(df_range=df_range, new_samples=extra_n, random_state=random_state, k_neighbors=k_neighbors)
            df_range_balanced = pd.concat([df_range, df_synthetic],ignore_index=True)

        else:
            df_range_balanced = df_range

        balanced_parts.append(df_range_balanced)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1,random_state=random_state).reset_index(drop=True)
    df_balanced = df_balanced.drop(columns=["glycemic_range"])

    # Add distribution summary after balancing

    return df_balanced

def _generate_smoter_samples(df_range: pd.DataFrame, new_samples: int, random_state: int = 42, k_neighbors: int = 10) -> pd.DataFrame:
    # Patient-aware
    min_samples_per_patient = 2
    random_generator = np.random.default_rng(random_state)

    # Patients eligible for SMOTER in this specific glycemic range
    patient_counts = df_range.groupby("patient_id").size()
    eligible_counts = patient_counts[patient_counts >= min_samples_per_patient]

    # If no patient has enough samples, fallback to random
    if eligible_counts.empty:
        return df_range.sample(n=new_samples, replace=True, random_state=random_state,).reset_index(drop=True)

    # Allocate new synthetic samples proportionally to each patient's contribution
    raw_allocation = new_samples * eligible_counts / eligible_counts.sum()
    patient_allocation = {
        patient_id: int(value)
        for patient_id, value in raw_allocation.items()
    }

    # Allocate rounding remainders to the patient with the most samples
    missing = new_samples - sum(patient_allocation.values())
    if missing > 0:
        main_patient = eligible_counts.idxmax()
        patient_allocation[main_patient] += missing

    feature_cols = _get_feature_columns(df_range)
    synthetic_rows = []

    for patient_id, patient_new_samples in patient_allocation.items():

        if patient_new_samples == 0:
            continue

        df_patient = df_range[df_range["patient_id"] == patient_id].reset_index(drop=True)
        values = df_patient[feature_cols].to_numpy(dtype=float)

        # SMOTER implementation
        n_neighbors = min(k_neighbors + 1, len(df_patient))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(values)

        neighbor_indices = nn.kneighbors(values, return_distance=False)

        for _ in range(patient_new_samples):
            row_pos = random_generator.integers(0, len(df_patient))

            possible_neighbors = neighbor_indices[row_pos][1:]

            if len(possible_neighbors) == 0:
                synthetic_rows.append(df_patient.iloc[row_pos].copy())
                continue

            neighbor_pos = random_generator.choice(possible_neighbors)

            row = df_patient.iloc[row_pos].copy()

            row_values = df_patient.iloc[row_pos][feature_cols].astype(float)
            neighbor_values = df_patient.iloc[neighbor_pos][feature_cols].astype(float)

            alpha = random_generator.random()

            synthetic_values = row_values + alpha * (neighbor_values - row_values)
            # ToDop: int????
            row[feature_cols] = np.rint(synthetic_values).astype(int)

            synthetic_rows.append(row)

    df_synthetic = pd.DataFrame(synthetic_rows).reset_index(drop=True)

    return df_synthetic