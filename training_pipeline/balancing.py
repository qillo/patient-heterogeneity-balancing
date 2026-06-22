import pandas as pd

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
        print("SMOTER is not implemented yet.")

    elif method == "smogn":
        print("SMOGN is not implemented yet.")

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