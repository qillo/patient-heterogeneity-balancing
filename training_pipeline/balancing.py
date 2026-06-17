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


def balance_training_set(
    df_train: pd.DataFrame,
    method: str,
    level: str
) -> pd.DataFrame:

    if method not in ["random", "smoter", "smogn"]:
        raise ValueError("method must be: random, smoter or smogn")

    if level == "full":
        target_proportions = FULL_TARGET_PROPORTIONS
    elif level == "semi":
        target_proportions = SEMI_TARGET_PROPORTIONS
    else:
        raise ValueError("level must be: full or semi")

    if method == "random":
        print("SMOTER is not implemented yet.")

    elif method == "smoter":
        print("SMOTER is not implemented yet.")

    elif method == "smogn":
        print("SMOGN is not implemented yet.")