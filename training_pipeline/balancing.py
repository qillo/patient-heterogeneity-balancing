import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

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
def add_glycemic_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        df["y"] < 54,
        (df["y"] >= 54) & (df["y"] < 70),
        (df["y"] >= 70) & (df["y"] < 181),
        (df["y"] >= 181) & (df["y"] < 251),
        df["y"] >= 251,
    ]

    choices = [
        "severe_hypoglycemia",
        "hypoglycemia",
        "normoglycemia",
        "hyperglycemia",
        "severe_hyperglycemia",
    ]

    df["glycemic_range"] = np.select(
        conditions,
        choices,
        default="unknown"
    )

    return df

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

def _get_patient_allocation(df_range: pd.DataFrame, new_samples: int, min_samples_per_patient: int = 2) -> dict:
    patient_counts = df_range.groupby("patient_id").size()
    eligible_counts = patient_counts[patient_counts >= min_samples_per_patient]

    if eligible_counts.empty:
        return {}

    raw_allocation = new_samples * eligible_counts / eligible_counts.sum()
    patient_allocation = {
        patient_id: int(value)
        for patient_id, value in raw_allocation.items()
    }

    missing = new_samples - sum(patient_allocation.values())
    if missing > 0:
        main_patient = eligible_counts.idxmax()
        patient_allocation[main_patient] += missing

    return patient_allocation

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
        return _smogn_balance(df_train, target_proportions)

### DELEGATED FUNCTIONS

# ABSTRACT BALANCING FUNCTION
def _balance_by_range(df_train: pd.DataFrame, target_proportions: dict, oversampler, random_state: int = 42, **oversampler_kwargs) -> pd.DataFrame:

    start_time = time.time()
    print(f"\n[[[BALANCE]]] Starting balance | rows_before={len(df_train)}", flush=True)

    df_train = add_glycemic_range(df_train)

    total_target = len(df_train)
    target_counts = get_target_counts(total_target, target_proportions)

    feature_cols = _get_feature_columns(df_train)

    balanced_parts = []

    for range_name, target_n in target_counts.items():
        df_range = df_train[df_train["glycemic_range"] == range_name]

        current_n = len(df_range)

        print(f"[[BALANCE]] Range={range_name} | current={current_n} | target={target_n}", flush=True)

        if current_n == 0:
            print(f"{range_name} has 0 samples. Skipping.")
            continue

        if current_n > target_n:
            print(f"[BALANCE] -> undersampling | removing={current_n - target_n}", flush=True)

            df_range_balanced = df_range.sample(n=target_n, replace=False, random_state=random_state)

        elif current_n < target_n:
            extra_n = target_n - current_n

            print(f"[BALANCE] -> oversampling | generating={extra_n}", flush=True)

            df_extra = oversampler(
                df_range=df_range,
                new_samples=extra_n,
                random_state=random_state,
                feature_cols=feature_cols,
                **oversampler_kwargs
            )

            df_range_balanced = pd.concat([df_range, df_extra], ignore_index=True)

        else:
            print(f"[BALANCE] -> unchanged", flush=True)
            df_range_balanced = df_range

        balanced_parts.append(df_range_balanced)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_balanced = df_balanced.drop(columns=["glycemic_range"])

    elapsed = time.time() - start_time
    print(f"[BALANCE] Finished balance | rows_after={len(df_balanced)} | time={elapsed:.2f}s\n", flush=True)

    return df_balanced

# RANDOM
def _random_balance(df_train: pd.DataFrame, target_proportions: dict, random_state: int = 42) -> pd.DataFrame:
    return _balance_by_range(
        df_train=df_train,
        target_proportions=target_proportions,
        oversampler=_generate_random_samples,
        random_state=random_state
    )

# SMOTER
def _smoter_balance(df_train: pd.DataFrame, target_proportions: dict, random_state: int = 42, k_neighbors: int = 5) -> pd.DataFrame:
    return _balance_by_range(
        df_train=df_train,
        target_proportions=target_proportions,
        oversampler=_generate_smoter_samples,
        random_state=random_state,
        k_neighbors=k_neighbors
    )

# SMOGN
def _smogn_balance(df_train: pd.DataFrame, target_proportions: dict, random_state: int = 42, k_neighbors: int = 5, noise_factor: float = 0.05) -> pd.DataFrame:
    return _balance_by_range(
        df_train=df_train,
        target_proportions=target_proportions,
        oversampler=_generate_smogn_samples,
        random_state=random_state,
        k_neighbors=k_neighbors,
        noise_factor=noise_factor
    )

# SAMPLING FUNCTIONS
def _generate_random_samples(df_range: pd.DataFrame, new_samples: int, random_state: int = 42, **kwargs) -> pd.DataFrame:
    return df_range.sample(n=new_samples, replace=True, random_state=random_state).reset_index(drop=True)

def _generate_smoter_samples(df_range: pd.DataFrame, new_samples: int, random_state: int = 42, k_neighbors: int = 10, feature_cols: list | None = None) -> pd.DataFrame:
    # Patient-aware
    random_generator = np.random.default_rng(random_state)
    patient_allocation = _get_patient_allocation(df_range, new_samples, min_samples_per_patient=2)

    if not patient_allocation:
        return _generate_random_samples(df_range, new_samples, random_state=random_state)

    if feature_cols is None:
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

def _generate_smogn_samples(df_range: pd.DataFrame, new_samples: int, random_state: int = 42, k_neighbors: int = 10, noise_factor: float = 0.05, feature_cols: list | None = None) -> pd.DataFrame:
    # Patient-aware
    random_generator = np.random.default_rng(random_state)
    patient_allocation = _get_patient_allocation(df_range, new_samples, min_samples_per_patient=2)

    if not patient_allocation:
        return _generate_random_samples(df_range, new_samples, random_state=random_state)

    if feature_cols is None:
        feature_cols = _get_feature_columns(df_range)

    synthetic_rows = []

    for patient_id, patient_new_samples in patient_allocation.items():

        if patient_new_samples == 0:
            continue

        df_patient = df_range[df_range["patient_id"] == patient_id].reset_index(drop=True)
        values = df_patient[feature_cols].to_numpy(dtype=float)

        # SMOGN implementation
        n_neighbors = min(k_neighbors + 1, len(df_patient))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(values)

        distances, neighbor_indices = nn.kneighbors(values, return_distance=True)

        # Distance threshold: close neighbors use SMOTER, far neighbors use Gaussian noise
        valid_distances = distances[:, 1:].ravel()
        distance_threshold = np.median(valid_distances)

        feature_std = df_patient[feature_cols].std().fillna(0).to_numpy(dtype=float)

        for _ in range(patient_new_samples):
            row_pos = random_generator.integers(0, len(df_patient))

            possible_neighbors = neighbor_indices[row_pos][1:]
            possible_distances = distances[row_pos][1:]

            if len(possible_neighbors) == 0:
                synthetic_rows.append(df_patient.iloc[row_pos].copy())
                continue

            selected_idx = random_generator.integers(0, len(possible_neighbors))
            neighbor_pos = possible_neighbors[selected_idx]
            neighbor_distance = possible_distances[selected_idx]

            row = df_patient.iloc[row_pos].copy()

            row_values = df_patient.iloc[row_pos][feature_cols].astype(float)

            # SMOTER case: interpolate with a close neighbor
            if neighbor_distance <= distance_threshold:
                neighbor_values = df_patient.iloc[neighbor_pos][feature_cols].astype(float)
                alpha = random_generator.random()

                synthetic_values = row_values + alpha * (neighbor_values - row_values)

            # SMOGN case: add Gaussian noise around the selected real sample
            else:
                noise = random_generator.normal(loc=0, scale=feature_std * noise_factor, size=len(feature_cols))

                synthetic_values = row_values + noise

            row[feature_cols] = np.rint(synthetic_values).astype(int)

            synthetic_rows.append(row)

    df_synthetic = pd.DataFrame(synthetic_rows).reset_index(drop=True)

    return df_synthetic