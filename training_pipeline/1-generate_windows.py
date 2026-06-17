import numpy as np
import pandas as pd
import os

# Windows creation parameters -----------------------
history_length = 8  # 120 min
horizons = [2, 4, 6, 8]  # 2 for 30 min, 4 for 60 min
# horizons = [2]
deduplicate_intra_patient = True  # Remove duplicated instances in a patient

path_to_load_data = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/Datasets/2025-12-05/Preprocessed/diatrend/Final_versions_V4/Filtered_for_prediction/Glucose_measurements_FILTERED_2025-11-20.parquet'
path_to_save_windows = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/windows/Extra_Fields/'

dataset_name = "DiaTrend"

# Windows creation parameters END -----------------------

if not os.path.exists(path_to_save_windows):
    # If it does not exist, create it
    os.makedirs(path_to_save_windows)

# Load the dataset
df_data = pd.read_parquet(path_to_load_data)

# Filter valid rows (15min is not NaT)
df_valid_rows = df_data[df_data['15min'].notna()]

# Sort columns by 'Patient_ID' and '15min'
df_valid_rows = df_valid_rows.sort_values(by=['Patient_ID', '15min'])
df_valid_rows.reset_index(drop=True, inplace=True)  # Reset index after filtering

# Group by 'Patient_ID' and convert 'Measurement' to numpy array
patient_dict = {
    pid: group['Measurement'].to_numpy()
    for pid, group in df_valid_rows.groupby('Patient_ID')
}

# Group by 'Patient_ID' and convert '15min' to numpy array of dates
date_dict = {
    pid: group['15min'].dt.date.to_numpy()
    for pid, group in df_valid_rows.groupby('Patient_ID')
}

# Group by 'Patient_ID' and convert '15min' to numpy array of times
time_dict = {
    pid: group['15min'].dt.time.to_numpy()
    for pid, group in df_valid_rows.groupby('Patient_ID')
}

print(f"Number of patients with valid data: {len(patient_dict)}")

# Auxiliary function
def get_windows_one_step_walk_forward(bgl_measurement_dict,
                                      history_length,
                                      horizon,
                                      deduplicate_intra_patient) -> pd.DataFrame:
    """
      :param bgl_measurement_dict: Dictionary with patient_id as key and numpy array of temporal series of BGL measurements
      :param history_length: Number of samples used to predict (8 for 120 minutes)
      :param horizon: Prediction horizon in number of windows (2 for 30 minutes, 4 for 60 minutes, and so on)
      :return: Get windows without missing values from one step sliding windows
    """
    df_all_patient_set = pd.DataFrame()

    for patient_id, patient_series in bgl_measurement_dict.items():

        # Creating the one step sliding window for BGL measurements, dates, and times.
        # Input windows (x)
        x = np.lib.stride_tricks.sliding_window_view(patient_series[:-horizon], history_length)
        x_date = np.lib.stride_tricks.sliding_window_view(date_dict[patient_id][:-horizon], history_length)
        x_time = np.lib.stride_tricks.sliding_window_view(time_dict[patient_id][:-horizon], history_length)

        # Output windows (y)
        y = np.lib.stride_tricks.sliding_window_view(patient_series[history_length:], horizon)

        # Removing rows with missing values (NaN)
        nan_rows_x = np.isnan(x).any(axis=1)  # In input values
        nan_rows_y = np.isnan(y).any(axis=1)  # In output values
        x = x[~(nan_rows_x | nan_rows_y)]  # Remove rows with NaN in either x or y from x
        x_date = x_date[~(nan_rows_x | nan_rows_y)]  # Remove rows with NaN from x_date
        x_time = x_time[~(nan_rows_x | nan_rows_y)]  # Remove rows with NaN from x_time

        y = y[~(nan_rows_x | nan_rows_y)]  # Remove rows with NaN in either x or y from y

        # Create DataFrame for the current patient
        df_x = pd.DataFrame(x)  # Input windows
        # Round to 1 decimal place to avoid floating point issues when comparing for duplicates. This is especially
        # important if the original measurements have decimal values, as it can help to reduce the number of unique
        # values and increase the chances of finding duplicates.
        df_x = df_x.round(1)
        # Get only the last column of x_date, which corresponds to the date of the last measurement in the input window.
        # This is because we want to associate each input window with a single date, which will be the date of the last
        # measurement in the window. The same applies to x_time.
        df_x_date = pd.DataFrame(x_date[:, -1])
        df_x_time = pd.DataFrame(x_time[:, -1])  # Input windows times

        df_y = pd.DataFrame(y[:, -1])  # Output windows (last column of y)
        # Round to 1 decimal place to avoid floating point issues when comparing for duplicates. (Idem as for df_x)
        df_y = df_y.round(1)

        # Concatenate all the DataFrames horizontally (inputs, outputs, dates, and times).
        df_set = pd.concat([df_x, df_y, df_x_date, df_x_time], axis=1)

        # Renaming columns and adding patient_id:
        # Input: x0 a x7, Output: y, Date: x_date_{last column of x}, Time: x_time_{last column of x}. The date and time
        # columns are named with the last column of x to indicate that they correspond to the date and time of the last
        # measurement in the input window.
        new_cols = [f"x{i}" for i in range(8)] + ["y"] + [f"x_date_{x.shape[1] - 1}"] + [f"x_time_{x.shape[1] - 1}"]
        df_set.columns = new_cols
        df_set['patient_id'] = patient_id

        # Concat the current patient's DataFrame to the main DataFrame
        df_all_patient_set = pd.concat([df_all_patient_set, df_set], ignore_index=True)

    if deduplicate_intra_patient:
        # Drop duplicated instances in a patient. All input & output columns are considered for duplication, therefore,
        # if a patient has the same input and output values, it will be considered a duplicate, but if there are two
        # equal instances but from different patients, they will not be considered duplicates.
        print("Deduplicating intra-patient instances...")
        print(f"Initial DataFrame shape: {df_all_patient_set.shape}")

        # Search for duplicated rows within each patient_id group and mark them as duplicates but only on the input and
        # output columns (x0 to x7 and y), ignoring the date and time columns for duplication. This way, if two rows
        # have the same input and output values but different dates or times, they will be considered duplicates.
        df_all_patient_set['is_duplicate'] = df_all_patient_set.duplicated(subset=[f"x{i}" for i in range(8)] + ["y"] + ['patient_id'],
                                                                           keep='first')
        df_all_patient_set = df_all_patient_set[~df_all_patient_set['is_duplicate']]  # Keep only non-duplicate rows
        df_all_patient_set = df_all_patient_set.drop(columns=['is_duplicate'])  # Drop the helper column
        df_all_patient_set.reset_index(drop=True, inplace=True)  # Reset index after dropping duplicates
        print(f"DataFrame shape after intra-patient deduplication: {df_all_patient_set.shape}")

    return df_all_patient_set


for current_horizon in horizons:
    print(f"Processing horizon: {current_horizon}")
    df_windows = get_windows_one_step_walk_forward(bgl_measurement_dict=patient_dict,
                                                   history_length=history_length,
                                                   horizon=current_horizon,
                                                   deduplicate_intra_patient=deduplicate_intra_patient)

    # NOTE: Replace with your filepath
    df_windows.to_parquet(f'{path_to_save_windows}/windows_horizon_{dataset_name}_{current_horizon}.parquet')
    print(f"Saved windows for horizon {current_horizon} to parquet files.")
    print("--------------------------------------------------")
    print()
