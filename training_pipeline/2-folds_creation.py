# This script creates patient-wise cross-validation folds for a dataset of T1D windows.
# The split proportions are approximately 70% train, 10% validation, and 20% test.
# Current script is design to make 5 folds

import pandas as pd
import os
import random
from collections import Counter
from typing import Dict, List

# Type alias for clarity: each fold is a dict with three lists of patient IDs (str)
Fold = Dict[str, List[str]]  # {'train': [...], 'val': [...], 'test': [...]}

dataset_name = "DiaTrend"
path_read_data = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/windows/Extra_Fields/'
path_to_save_folds = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/windows/Extra_Fields/folds/'
horizon_list = [2, 4, 6, 8]



# Functions -------------------------
def make_folds_greedy_splits_valanced_priority(patient_windows: Dict[str, int],
                                               k: int = 5,
                                               seed: int = 42,
                                               ) -> List[Fold]:
    """Balanced *k*-fold split (patient‑wise).

    • **Test assignment (unchanged)** – patients are sorted globally by window
      count ↓, then taken in *k*-sized blocks; one patient from each block is
      randomly assigned to each test fold so that big patients are spread
      evenly.  Remaining (< *k*) patients go to the fold with the lightest
      load.

    • **Train / Val assignment (NEW)** – the remaining patients of each fold
      are also processed in blocks of *k* **(8 → 7‑train / 1‑val)**.  The last
      block, even if smaller than *k*, still sends **one random patient to
      validation** and the rest to training.  This guarantees *at least one*
      validation patient per fold.
    """

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1️⃣  Sort all patients by window count (descending)
    # ------------------------------------------------------------------
    sorted_patients = sorted(patient_windows.keys(),
                             key=lambda pid: patient_windows[pid],
                             reverse=True)

    # ------------------------------------------------------------------
    # 2️⃣  Assign test patients block‑wise (size == k)
    # ------------------------------------------------------------------
    test_folds: List[List[str]] = [[] for _ in range(k)]
    fold_loads: List[int] = [0] * k

    block_size = k  # 5 if k == 5 (one per fold)
    idx = 0
    while idx + block_size <= len(sorted_patients):
        block = sorted_patients[idx: idx + block_size]
        block.sort(key=lambda pid: patient_windows[pid], reverse=True)
        rng.shuffle(block)  # random perm in the block
        for fold_idx, patient_id in enumerate(block):
            test_folds[fold_idx].append(patient_id)
            fold_loads[fold_idx] += patient_windows[patient_id]
        idx += block_size

    # Remaining (< k) patients – assign to lightest fold
    for pid in sorted_patients[idx:]:
        lightest = fold_loads.index(min(fold_loads))
        test_folds[lightest].append(pid)
        fold_loads[lightest] += patient_windows[pid]

    # ------------------------------------------------------------------
    # 3️⃣  Build train / val with 7‑train / 1‑val blocks (size == 8)
    # ------------------------------------------------------------------
    all_ids_set = set(patient_windows.keys())
    folds: List[Fold] = []
    train_block = 7  # patients for train per block of 8
    val_block = 1  # patients for val  per block of 8
    block_tv = train_block + val_block  # 8

    for fold_idx in range(k):
        test_ids = set(test_folds[fold_idx])
        remaining_ids = [pid for pid in sorted_patients if pid not in test_ids]

        # --- process in blocks of 8 ------------------------------------
        train_ids, val_ids = [], []
        ptr = 0
        n_remaining = len(remaining_ids)

        while ptr + block_tv <= n_remaining:
            block = remaining_ids[ptr: ptr + block_tv]
            # already sorted globally; still shuffle to randomise who goes to val
            rng.shuffle(block)
            val_ids.append(block[0])  # 1 patient → val
            train_ids.extend(block[1:])  # 7 patients → train
            ptr += block_tv

        # Handle last (incomplete) block → send 1 to val, rest to train
        tail = remaining_ids[ptr:]
        if tail:
            rng.shuffle(tail)
            val_ids.append(tail[0])
            train_ids.extend(tail[1:])

        folds.append({
            'train': sorted(train_ids),
            'val': sorted(val_ids),
            'test': sorted(test_folds[fold_idx])
        })

    return folds

def validate_folds(folds: List[Fold], patient_measurements: Dict[str, int], tolerance_pct: float = 10.0,
                   horizon: int = 2) -> None:
    """
    Validate a list of patient-wise cross-validation folds.

    Parameters
    ----------
    folds : list[Fold]
        Output of `make_folds_greedy_splits` or similar.  Each element is a
        dictionary with keys ``'train'``, ``'val'``, ``'test'`` mapping to lists
        of patient IDs **(strings)**.

    patient_measurements : dict[str, int]
        Mapping ``{patient_id: number_of_measurements}``.  Used to check load
        balancing (Rule R4).

    tolerance_pct : float, default 10.0
        Maximum allowed percentage deviation from the mean test load before
        a warning is raised (Rule R4).

    horizon : int, default 2
        The horizon used for the folds, included in the printed messages.

    Raises
    ------
    AssertionError
        If any of the mandatory rules (R1–R3) is violated.

    Side effects
    ------------
    * Prints a success message for each rule that passes.
    * Prints the test-set load per fold and the percentage deviation from the
      mean (always), and a warning if any fold exceeds *tolerance_pct*.

    Notes
    -----
    *R4* (load balance) is **informative**; it does **not** raise an error.
    Adjust *tolerance_pct* as appropriate for your project.
    """

    print(f"\nValidating folds for horizon: {horizon}...")

    all_patients = set(patient_measurements)

    # ------------------------------------------------------------------ R1/R3
    for idx, fold in enumerate(folds):
        tr, va, te = map(set, (fold['train'], fold['val'], fold['test']))

        # R1 – exclusivity within the same fold
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), (
            f"[R1] Overlap detected in fold {idx}"
        )

        # R3 – union covers the entire cohort
        assert len(tr | va | te) == len(all_patients), (
            f"[R3] Missing patients in fold {idx}"
        )

    print("\n ✓ R1 and R3 passed for every fold.")

    # ------------------------------------------------------------------ R2
    test_counter = Counter(pid for fold in folds for pid in fold['test'])
    duplicates = [pid for pid, cnt in test_counter.items() if cnt != 1]

    assert not duplicates, (
        f"[R2] Patients appearing in test 0 or >1 times: {duplicates}"
    )
    print("✓ R2 passed – each patient appears in *test* exactly once.")

    # ------------------------------------------------------------------ R4 (informative)
    def test_load(ids: List[str]) -> int:
        """Total number of measurements for this group of patients."""
        return sum(patient_measurements[pid] for pid in ids)

    loads = [test_load(f['test']) for f in folds]
    mean_load = sum(loads) / len(loads)
    pct_dev = [100 * (l - mean_load) / mean_load for l in loads]

    print("\nTest-set load per fold (number of measurements):", loads)
    print("Deviation from mean (%)                      :", [round(p, 2) for p in pct_dev])

    if any(abs(p) > tolerance_pct for p in pct_dev):
        print(f"⚠️  [R4] Warning: at least one fold deviates by more than {tolerance_pct}% "
              "from the mean test load.")
    else:
        print(f"✓ R4 passed – test loads within ±{tolerance_pct}% of the mean.")

    print(f"\nAll mandatory validation checks for horizon {horizon} completed successfully.")


def tag_folds_and_save(
        df_windows: pd.DataFrame,
        folds: List[Fold],
        output_path: str,
        filename: str = "windows_with_5folds_",
        horizon: int = 2,
        column_prefix: str = "fold") -> None:
    """
    Add one column per fold (fold0 … fold{k-1}) indicating whether each window
    belongs to 'train', 'val' or 'test' in that fold, then write a single
    Parquet file.

    Parameters
    ----------
    df_windows : pd.DataFrame
        Original window table (must include column 'patient_id').

    folds : list[Fold]
        Output from make_folds_greedy_splits (patient IDs per split).

    output_path : str
        Destination filename, e.g. 'windows_with_folds.parquet'.

    column_prefix : str, default "fold"
        Prefix for the new columns: fold0, fold1, …

    parquet_kwargs : dict, optional
        Extra arguments passed to DataFrame.to_parquet.
    """
    df_tagged = df_windows.copy()

    for idx, fold in enumerate(folds):
        # Build a mapping patient_id -> split label for this fold
        label_map = {pid: "train" for pid in fold["train"]}
        label_map.update({pid: "val" for pid in fold["val"]})
        label_map.update({pid: "test" for pid in fold["test"]})

        col_name = f"{column_prefix}{idx}"
        df_tagged[col_name] = df_tagged["patient_id"].map(label_map).astype("category")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_tagged.to_parquet(f'{output_path}{filename}.parquet', index=False)
    print(f"\n ✓ File {filename}_horizon_{horizon} written: {output_path}"
          f"({len(df_windows):,} rows, {len(folds)} fold-columns)")

    return df_tagged


for horizon in horizon_list:
    print(f"\nReading data file from {path_read_data} for horizon {horizon}...")
    df_windows = pd.read_parquet(f"{path_read_data}windows_horizon_{dataset_name}_{horizon}.parquet")
    unique_patient_ids = df_windows['patient_id'].unique()
    num_patients = len(unique_patient_ids)

    # Count the number of windows per patient and convert to a dictionary
    patient_number_windows_dict = (
        df_windows
        .groupby("patient_id")  # group all rows that share the same ID
        .size()  # count rows per group
        .astype(int)  # ensure Python int, not numpy int64
        .to_dict()  # ==> { 'ID_1': n1, 'ID_2': n2, ... }
    )

    # Create folds using the greedy split function
    folds_split = make_folds_greedy_splits_valanced_priority(patient_number_windows_dict,
                                                             seed=42)

    # Print statistics of the folds created
    print(f'\n ------- Statistics of folds created for horizon: {horizon}-------')
    for i, fold in enumerate(folds_split):
        print(f"\n---- Fold {i + 1}: -----")
        print("-- Totals --")
        print(
            f"  Train: {len(fold['train'])} patients. Total windows: {sum(patient_number_windows_dict[pid] for pid in fold['train'])}")
        print(
            f"  Val:   {len(fold['val'])} patients. Total windows: {sum(patient_number_windows_dict[pid] for pid in fold['val'])}")
        print(
            f"  Test:  {len(fold['test'])} patients. Total windows: {sum(patient_number_windows_dict[pid] for pid in fold['test'])}")

        print("\n -- Percentages --")
        print(
            f"  Train: {(len(fold['train']) / num_patients * 100):.2f} percentage of patients. Percentage of windows: {(sum(patient_number_windows_dict[pid] for pid in fold['train']) / sum(patient_number_windows_dict.values()) * 100):.2f}%")
        print(
            f"  Val:   {(len(fold['val']) / num_patients * 100):.2f} percentage of patients. Percentage of windows: {(sum(patient_number_windows_dict[pid] for pid in fold['val']) / sum(patient_number_windows_dict.values()) * 100):.2f}%")
        print(
            f"  Test:  {(len(fold['test']) / num_patients * 100):.2f} percentage of patients. Percentage of windows: {(sum(patient_number_windows_dict[pid] for pid in fold['test']) / sum(patient_number_windows_dict.values()) * 100):.2f}%")

    print(f'\nEnd of statistics for folds created for horizon: {horizon}')

    # Validate the folds created
    validate_folds(folds=folds_split, patient_measurements=patient_number_windows_dict, horizon=horizon)

    # save_folds_as_parquet(df_windows, folds_split, "c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/folds/")

    # Save the folds as Parquet files
    tag_folds_and_save(df_windows=df_windows,
                       folds=folds_split,
                       output_path=path_to_save_folds,
                       filename=f'windows_with_folds_{dataset_name}_PH{horizon}',
                       horizon=horizon,
                       column_prefix='fold_')
