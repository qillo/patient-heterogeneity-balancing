# -*- coding: utf-8 -*-
"""
RELIEF-T1D — Training & Prediction Script (English)

Directory layout per (loss, algo, PH):
models/
  <prefix>.weights.h5
  <prefix>_history.csv
  <prefix>.best.json
  <prefix>_logs/  # TensorBoard (tfevents)
plots/
  training/
    <prefix>_Loss.pdf
results/
  predictions/
    df_test_results_vectors_<prefix>.parquet
    df_test_results_vectors_<loss>_<algo>_H<PH>_ALL.parquet
  evaluation/
    metrics/
      metrics_performance_by_range_<loss>_<algo>_H<PH>_ALL.csv
      number_of_points_by_zones_<loss>_<algo>_H<PH>_ALL.csv
      predictions_out_of_limits_<loss>_<algo>_H<PH>_ALL.csv
    figures/
      Clarke_Error_Grid_<...>_ALL.png

Each experiment directory contains a self-contained log file (experiment_log.out).
No global log file is created.
"""
# ----------------------- Imports -----------------------
from test_predictions import test_by_range, get_train_plots_loss, fold_results_aggregation
from loss_functions import RMSE, ClinicalPenaltyMetric

from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras

import numpy as np
import tensorflow as tf
import random
import os
import sys
import pandas as pd
import time
import datetime
import platform
import socket
import psutil
import subprocess
import json
import re
# ----------------------- End imports -----------------------

# ----------------------- Reproducibility -----------------------
np.random.seed(50)
tf.random.set_seed(50)
random.seed(50)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
# ----------------------- End reproducibility -----------------------

# ----------------------- Config -----------------------
batch_size = 4096
patience = 10
max_epoch = 5
dropout = 0.1
recurrent_dropout = 0.0
do_cross_validation = True

dataset_name = '_FILL_'
windows_path = '_FILL_'
history_length = 8
loss_function = RMSE
loss_function_name= 'RMSE'
horizon = 4
algorithm = 'LSTM'

maximum_sensor_reading = 401 # Diatrend and REPLACE-BG
# maximum_sensor_reading = 500 # T1DiabetesGranada
minimum_sensor_reading = 39 # Diatrend and REPLACE-BG
# minimum_sensor_reading = 40 # T1DiabetesGranada

# ----------------------- End Config -----------------------

# ----------------------- Console Tee Logic -----------------------
class Tee:
    def __init__(self, *files):
        self.files = [f for f in files if f]

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                try:
                    f.write(obj)
                    f.flush()
                except (IOError, ValueError):
                    pass

    def flush(self):
        for f in self.files:
            if not f.closed:
                try:
                    f.flush()
                except (IOError, ValueError):
                    pass

RUN_TS = datetime.datetime.now()
GLOBAL_RUN_ID = RUN_TS.strftime("%Y%m%d_%H%M%S")
original_stdout = sys.stdout
original_stderr = sys.stderr
# ----------------------- End Console Tee Logic -----------------------

# ----------------------- System & Git helpers -----------------------
def print_system_info():
    print("📅 Date & Time")
    print("-----------------------------------")
    print(f"Script Execution Start: {RUN_TS.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Time: {datetime.datetime.now()}")
    print("\n🖥️  System & Hardware")
    print("-----------------------------------")
    print(f"Hostname         : {socket.gethostname()}")
    print(f"OS               : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture     : {platform.machine()}")
    print(f"CPU              : {platform.processor()}")
    try:
        phys = psutil.cpu_count(logical=False)
    except Exception:
        phys = None
    print(f"CPU Cores        : {phys} physical / {psutil.cpu_count()} logical")
    ram = psutil.virtual_memory()
    print(f"RAM Total        : {round(ram.total / (1024 ** 3), 2)} GB")
    print(f"RAM Available    : {round(ram.available / (1024 ** 3), 2)} GB")
    print(f"RAM Used         : {round(ram.used / (1024 ** 3), 2)} GB")
    print(f"RAM Usage (%)    : {ram.percent}%")
    print("\n🐍 Python & Libraries")
    print("-----------------------------------")
    print(f"Python Version   : {platform.python_version()}")
    print(f"Executable Path  : {sys.executable}")
    print(f"TensorFlow       : {tf.__version__}")
    print("\n🧠 GPU (via TensorFlow)")
    print("-----------------------------------")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}           : {getattr(gpu, 'name', str(gpu))}")
        else:
            print("No GPUs detected by TensorFlow.")
    except Exception as e:
        print(f"GPU detection error: {e}")
    print("\n🎮 GPU Details (via nvidia-smi)")
    print("-----------------------------------")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT
        )
        for i, line in enumerate(out.decode(errors="ignore").strip().split("\n")):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name, mem_total, mem_used, util, driver = parts[:5]
                print(f"GPU {i}           : {name}")
                print(f"  Total Memory   : {mem_total}")
                print(f"  Used Memory    : {mem_used}")
                print(f"  Utilization    : {util}")
                print(f"  Driver Version : {driver}")
    except Exception as e:
        print(f"nvidia-smi not available: {e}")


def get_git_info():
    info = {"commit": "N/A", "branch": "N/A", "remote": "N/A"}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        info["remote"] = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
    except Exception:
        pass
    return info
# ----------------------- End system & git helpers -----------------------

# ----------------------- Experiment directory helpers -----------------------
def safe_token(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))

def safe_relpath(p, start=None):
    try:
        return os.path.relpath(p, start=start) if start else os.path.relpath(p)
    except Exception:
        return p

def next_versioned_dir(base_name: str) -> str:
    rx = re.compile(r"^(.*?)(?:-v(\d+))?$")
    m = rx.match(base_name)
    stem, v = m.group(1), m.group(2)
    vnum = int(v) if v else 1
    while True:
        candidate = f"{stem}-v{vnum}"
        if not os.path.exists(candidate):
            return candidate
        vnum += 1

def build_experiment_dir(algo: str, loss_name: str, ph_value, history_len: int, context: str) -> str:
    date_str = RUN_TS.strftime("%Y%m%d")
    base = f"EXP-{date_str}-prediction-{safe_token(loss_name)}-{safe_token(algo)}-H{history_len}-PH{safe_token(ph_value)}-{safe_token(context)}"
    exp_dir = next_versioned_dir(base)
    for sub in ["models", "plots/training", "results/predictions", "results/evaluation/metrics", "results/evaluation/figures", "summaries", "readmes"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir
# ----------------------- End experiment directory helpers -----------------------

# ----------------------- Model -----------------------
class LSTMModel:
    def __init__(self, input_shape, nb_output_units, nb_hidden_units=128,
                 dropout_rate=dropout, recurrent_dropout_rate=recurrent_dropout):
        self.input_shape, self.nb_output_units, self.nb_hidden_units = input_shape, nb_output_units, nb_hidden_units
        self.dropout, self.recurrent_dropout = dropout_rate, recurrent_dropout_rate
    def __repr__(self): return f"LSTM_{self.nb_hidden_units}_h_drop{self.dropout}_recDrop{self.recurrent_dropout}"
    def build(self):
        i = Input(shape=self.input_shape)
        x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(i)
        x = Dense(self.nb_output_units, activation=None)(x)
        return Model(inputs=[i], outputs=[x])

def build_LSTM_model(weights=''):
    model = LSTMModel(input_shape=(history_length, 1), nb_output_units=1)
    m = model.build()
    m.compile(loss=loss_function, optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyMetric()])
    if weights:
        print(f"Loading weights from: {weights}")
        m.load_weights(weights)
    return m
# ----------------------- End models -----------------------

# ----------------------- Keras callbacks -----------------------
def make_callbacks(filepath_prefix: str, early_stopping_patience: int):
    return [
        ModelCheckpoint(filepath=f"{filepath_prefix}.weights.h5", monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience, restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"{filepath_prefix}_logs", histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', profile_batch=0),
        keras.callbacks.CSVLogger(filename=f"{filepath_prefix}_history.csv", separator=',', append=False)
    ]
# ----------------------- End callbacks -----------------------

# ----------------------- Train function -----------------------
def train(x_train, y_train, x_val, y_val, model, history_len, save_prefix="", early_stopping_patience=patience):
    x_train = x_train.reshape(x_train.shape[0], history_len, 1)
    x_val = x_val.reshape(x_val.shape[0], history_len, 1)
    hist = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=max_epoch, callbacks=make_callbacks(save_prefix, early_stopping_patience))
    return hist, model
# ----------------------- End train function -----------------------

# ----------------------- Per-fold JSON/MD writers -----------------------
def write_fold_json_and_md(exp_dir, loss_name, algo, ph, fold_idx, fold_started_at, train_sec, train_hms, pred_sec, pred_hms, weights_path, hist_csv_path, best_json_path, preds_parquet_path, training_plot_prefix, history_len, do_cv, min_sensor, max_sensor, data_file_path, data_counts):
    payload = {
        "experiment": {"loss": loss_name, "algorithm": algo, "prediction_horizon": ph, "fold": fold_idx, "history_length": history_len, "cross_validation": do_cv, "sensor_limits": {"min": min_sensor, "max": max_sensor}, "started_at": fold_started_at.isoformat(), "run_id": GLOBAL_RUN_ID},
        "data": {"file": os.path.abspath(data_file_path), "counts": data_counts},
        "timing": {"train_seconds": round(train_sec, 3), "train_elapsed": str(train_hms), "predict_seconds": round(pred_sec, 3), "predict_elapsed": str(pred_hms)},
        "artifacts": {
            "weights_path": weights_path, "history_csv_path": hist_csv_path, "best_json_path": best_json_path,
            "predictions_parquet_path": preds_parquet_path, "training_plot_prefix": training_plot_prefix
        },
        "git": get_git_info(),
        "system": {"hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(), "tensorflow": tf.__version__}
    }
    
    for key, val in payload["artifacts"].items():
        payload["artifacts"][key] = safe_relpath(val, start=exp_dir)

    json_name = os.path.join(exp_dir, "summaries", f"summary_{loss_name}_{algo}_H{ph}_Fold{fold_idx}.json")
    with open(json_name, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)

    md_name = os.path.join(exp_dir, "readmes", f"README_{loss_name}_{algo}_H{ph}_Fold{fold_idx}.md")
    with open(md_name, "w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_idx} Summary for {loss_name}/{algo}/PH={ph}\n\n")
        f.write(f"**Data File**: `{os.path.abspath(data_file_path)}`\n\n")
        f.write("| Split | Patient Count | Window Count | % Patients | % Windows |\n")
        f.write("|-------|---------------|--------------|------------|-----------|\n")
        for split in ["train", "val", "test"]:
            f.write(f"| {split.capitalize()} | {data_counts['patients'][split]} | {data_counts['windows'][split]} | {data_counts['percent']['patients'][split]:.2f}% | {data_counts['percent']['windows'][split]:.2f}% |\n")
        f.write("\n## Timings\n")
        f.write(f"- **Training**: {train_hms}\n- **Prediction**: {pred_hms}\n\n")
        f.write("## Artifacts\n")
        for key, val in payload["artifacts"].items():
            f.write(f"- **{key.replace('_path', '').replace('_', ' ').capitalize()}**: `{val}`\n")
# ----------------------- End writers -----------------------

# ----------------------- Main README writer -----------------------
def write_experiment_readme(exp_dir, loss_name, algo, ph, history_len, context, k_folds):
    exp_name, git_info = os.path.basename(exp_dir), get_git_info()
    lines = [f"# Experiment Summary: `{exp_name}`\n",
             f"- **Run ID**: `{GLOBAL_RUN_ID}`", f"- **Date**: `{RUN_TS.strftime('%Y-%m-%d %H:%M:%S')}`",
             f"- **Git Commit**: `{git_info['commit']}`\n",
             "## 🧪 Experiment Configuration\n",
             f"- **Algorithm**: `{algo}`", f"- **Loss Function**: `{loss_name}`",
             f"- **Prediction Horizon (PH)**: `{ph}`", f"- **History Length (H)**: `{history_len}`\n",
             "## 📊 Quick Links\n",
             "- [TensorBoard Logs](./models/)", f"- [Aggregated Predictions](./results/predictions/df_test_results_vectors_{loss_name}_{algo}_H{ph}.parquet)",
             f"- [Aggregated Metrics](./results/evaluation/metrics/metrics_performance_by_range_{loss_name}_{algo}_H{ph}_ALL.csv)",
             "- [Clarke Error Grid (CEG) Plots](./results/evaluation/figures/)", "- [Detailed Log File](./experiment_log.out)\n",
             "---", f"\n## 📂 Fold Details ({k_folds} folds)\n"]
    for i in range(1, k_folds + 1):
        p = f"{loss_name}_{algo}_H{ph}_Fold{i}"
        lines.extend([f"### Fold {i}\n", f"- **Weights**: `models/{p}.weights.h5`", f"- **History**: `models/{p}_history.csv`",
                      f"- **Best Epoch**: `models/{p}.best.json`",
                      f"- **Predictions**: `results/predictions/df_test_results_vectors_{p}.parquet`",
                      f"- **Detailed Summary**: `readmes/README_{loss_name}_{algo}_H{ph}_Fold{i}.md`\n"])
    with open(os.path.join(exp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
# ----------------------- End Main README writer -----------------------

# ----------------------- Main Loop -----------------------
try:
    sys.stdout = Tee(original_stdout)
    print("Starting script execution...")

    # Redudant reproducibility setup
    tf.keras.backend.clear_session()
    np.random.seed(50)
    tf.random.set_seed(50)
    random.seed(50)

    used_columns = [f"x{i}" for i in range(history_length)] + ["y"]
    info_columns = ['patient_id', 'x_time_7', 'x_date_7']
    k_folds = 5 if do_cross_validation else 1
    
    '''
    datasets = ["DiaTrend", "REPLACE-BG", "T1DiabetesGranada"]
    balancing_methods = ["random", "smoter", "smogn"]
    balancing_levels = ["full", "semi"]

    for dataset_name, balancing_method, balancing_level in product(
        datasets,
        balancing_methods,
        balancing_levels
    ):

    experiment_context = f"{dataset_name}_{balancing_method}_{balancing_level}"
    '''

    EXP_DIR = build_experiment_dir(algorithm, loss_function_name, horizon, history_length, experiment_context)
    exp_log_file = None
    try:
        exp_log_file_path = os.path.join(EXP_DIR, "experiment_log.out")
        exp_log_file = open(exp_log_file_path, "a", encoding="utf-8")
        sys.stdout = sys.stderr = Tee(original_stdout, exp_log_file)
        
        print(f"================\nStarting Experiment: {os.path.basename(EXP_DIR)}\n================")
        print_system_info()
        print("-----------------------------------\n")

        MODELS_DIR = os.path.join(EXP_DIR, "models")
        PRED_DIR = os.path.join(EXP_DIR, "results", "predictions")
        PLOTS_TRAIN_DIR = os.path.join(EXP_DIR, "plots", "training")
        METRICS_DIR = os.path.join(EXP_DIR, "results", "evaluation", "metrics")
        FIGURES_DIR = os.path.join(EXP_DIR, "results", "evaluation", "figures")
        
        aggregated_results = []
        aggregated_results_performance = []
        aggregated_results_out_of_range = []
        aggregated_results_number_of_points = []

        for i in range(k_folds):
            current_fold = i + 1
            current_name = f"{loss_function_name}_{algorithm}_PH{horizon}_Fold{current_fold}"
            data_file_path = f"{windows_path}/windows_with_folds_{dataset_name}_PH{horizon}.parquet"

            print(f"\n---- Loading data for Fold {current_fold}/{k_folds} from: {data_file_path} ----")
            df = pd.read_parquet(data_file_path)

            df_train = df[df[f'fold_{i}'] == 'train']
            patients_train_number = df_train['patient_id'].nunique()

            '''
            train_windows_before = len(df_train)
            train_patients_before = df_train['patient_id'].nunique()

            # df_train = balance(df_train)

            train_windows_after = len(df_train)
            train_patients_after = df_train['patient_id'].nunique()
            '''

            df_train = df_train.reset_index(drop=True)[used_columns]
            
            df_val = df[df[f'fold_{i}'] == 'val']
            patients_val_number = df_val['patient_id'].nunique()
            df_val = df_val.reset_index(drop=True)[used_columns]

            df_test = df[df[f'fold_{i}'] == 'test']
            patients_test_number = df_test['patient_id'].nunique()
            df_test_info = df_test.reset_index(drop=True)[info_columns]
            df_test = df_test.reset_index(drop=True)[used_columns]
            
            total_patients = patients_train_number + patients_val_number + patients_test_number
            
            data_counts = {
                "patients": {"train": int(patients_train_number), "val": int(patients_val_number), "test": int(patients_test_number)},
                "windows": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
                "percent": {
                    "patients": {"train": (patients_train_number/total_patients*100), "val": (patients_val_number/total_patients*100), "test": (patients_test_number/total_patients*100)},
                    "windows": {"train": (len(df_train)/len(df)*100), "val": (len(df_val)/len(df)*100), "test": (len(df_test)/len(df)*100)}
                }
            }

            print("\n-- Data Split --")
            print(f"  Train: {patients_train_number} patients. Total windows: {len(df_train)}")
            print(f"  Val:   {patients_val_number} patients. Total windows: {len(df_val)}")
            print(f"  Test:  {patients_test_number} patients. Total windows: {len(df_test)}")

            x_train, y_train = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1:].to_numpy()
            x_val, y_val = df_val.iloc[:, :-1].to_numpy(), df_val.iloc[:, -1:].to_numpy()
            x_test, y_test = df_test.iloc[:, :-1].to_numpy(), df_test.iloc[:, -1:].to_numpy()
            del df, df_train, df_val, df_test  # Free memory

            print(f'\nSTART training {algorithm} - Fold={current_fold}')
            model = build_LSTM_model()

            fold_started_at = datetime.datetime.now()
            start_time = time.time()
            save_prefix = os.path.join(MODELS_DIR, current_name)
            hist, model_trained = train(x_train, y_train, x_val, y_val, model, history_length, save_prefix)
            train_time = time.time() - start_time
            print(f'END training {algorithm} - Fold={current_fold} (Time: {datetime.timedelta(seconds=train_time)})\n')
            
            print('START prediction & saving artifacts...')
            weights_file = f"{save_prefix}.weights.h5"
            model_load = build_LSTM_model(weights=weights_file)
            
            start_pred_time = time.time()
            if len(model_load.input_shape) == 3: x_test = x_test.reshape(x_test.shape[0], history_length, 1)
            y_pred = model_load.predict(x_test)
            pred_time = time.time() - start_pred_time

            # Concat x_test o patient ids and timestamps to y_test and y_pred, and save to parquet for later analysis
            df_res = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': y_pred.ravel()})
            df_res = pd.concat([df_res, df_test_info], axis=1)
            preds_path = os.path.join(PRED_DIR, f'df_test_results_vectors_{current_name}.parquet')
            df_res.to_parquet(preds_path)
            aggregated_results.append(df_res)

            # # Generate performance metrics for this fold
            # df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(df_res,
            #                                                 f'_{loss_function_name}_{algorithm}_H{horizon}_Fold{current_fold}',
            #                                                                               show_plot=True, plot_dir=FIGURES_DIR,
            #                                                                               minimum_sensor_reading=minimum_sensor_reading,
            #                                                                               maximum_sensor_reading=maximum_sensor_reading)
            # df_metrics_performance.round(2).to_csv(os.path.join(METRICS_DIR,
            #                                         f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_Fold{current_fold}.csv'),
            #                                        index=False)
            # df_number_of_points.to_csv(os.path.join(METRICS_DIR,
            #                              f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_Fold{current_fold}.csv'),
            #                            index=False)
            # df_out_of_limits.to_csv(os.path.join(METRICS_DIR,
            #                               f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_Fold{current_fold}.csv'),
            #                         index=False)
            #
            # aggregated_results_performance.append(df_metrics_performance)
            # aggregated_results_number_of_points.append(df_number_of_points)
            # aggregated_results_out_of_range.append(df_out_of_limits)

            training_plot_prefix = os.path.join(PLOTS_TRAIN_DIR, current_name)
            get_train_plots_loss(hist, training_plot_prefix)
            
            hist_csv_path = f"{save_prefix}_history.csv"
            best_json_path = f"{save_prefix}.best.json"
            
            try:
                df_hist = pd.read_csv(hist_csv_path)
                best_idx = int(df_hist['val_loss'].idxmin() if 'val_loss' in df_hist.columns else df_hist['loss'].idxmin())
                row = df_hist.iloc[best_idx].to_dict()
                best_payload = {
                    "monitor": "val_loss" if 'val_loss' in df_hist.columns else "loss",
                    "epoch_idx": best_idx, "epoch_count": int(df_hist.shape[0]), "metrics": row,
                    "artifacts": {"weights_path": os.path.abspath(weights_file), "history_csv_path": os.path.abspath(hist_csv_path)},
                    "context": {"loss": loss_function_name, "algorithm": algorithm, "prediction_horizon": horizon, "history_length": history_length, "fold": current_fold, "run_id": GLOBAL_RUN_ID},
                    "created_at": datetime.datetime.now().isoformat()
                }
                with open(best_json_path, "w", encoding="utf-8") as f: json.dump(best_payload, f, indent=2)
            except Exception as e:
                print(f"WARNING: could not write best.json for {current_name}: {e}")

            write_fold_json_and_md(
                EXP_DIR, loss_function_name, algorithm, horizon, current_fold,
                fold_started_at, train_time, datetime.timedelta(seconds=train_time),
                pred_time, datetime.timedelta(seconds=pred_time),
                weights_path=weights_file, hist_csv_path=hist_csv_path, best_json_path=best_json_path,
                preds_parquet_path=preds_path, training_plot_prefix=training_plot_prefix,
                history_len=history_length, do_cv=do_cross_validation,
                min_sensor=minimum_sensor_reading, max_sensor=maximum_sensor_reading,
                data_file_path=data_file_path, data_counts=data_counts
            )
            print(f'END prediction & saving artifacts for Fold={current_fold} (Time: {datetime.timedelta(seconds=pred_time)})\n')

        #------------------ Metric-Then-Aggregate approach ------------------

        fold_number = 1
        for df_current_result in aggregated_results:
            # Generate performance metrics for this fold

            print()
            print('Agregating metrics for fold ', fold_number)

            df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(df_current_result,
                                                                                            f'_{loss_function_name}_{algorithm}_H{horizon}_Fold{fold_number}',
                                                                                            show_plot=False,
                                                                                            plot_dir=FIGURES_DIR,
                                                                                            minimum_sensor_reading=minimum_sensor_reading,
                                                                                            maximum_sensor_reading=maximum_sensor_reading)
            df_metrics_performance.round(2).to_csv(os.path.join(METRICS_DIR,
                                                                f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_Fold{fold_number}.csv'),
                                                                index=False)
            df_metrics_performance['Fold'] = fold_number

            df_number_of_points.to_csv(os.path.join(METRICS_DIR,
                                                    f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_Fold{fold_number}.csv'),
                                                    index=False)
            df_number_of_points['Fold'] = fold_number

            df_out_of_limits.to_csv(os.path.join(METRICS_DIR,
                                                    f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_Fold{fold_number}.csv'),
                                    index=False)
            df_out_of_limits['Fold'] = fold_number

            aggregated_results_performance.append(df_metrics_performance)
            aggregated_results_number_of_points.append(df_number_of_points)
            aggregated_results_out_of_range.append(df_out_of_limits)

            print()
            print('Aggregated metrics for fold ', fold_number, ' done.')

            fold_number += 1

        print()
        print('Calculating aggregated metrics across folds (Metric-Then-Aggregate)...')

        # Calculate aggregated metrics across folds
        df_metrics_performance_agg = pd.concat(aggregated_results_performance, ignore_index=True)
        df_metrics_performance_agg.round(2).to_csv(os.path.join(METRICS_DIR, f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_ALL_metric_then_aggregate.csv'), index=False)
        df_metrics_performance_agg = df_metrics_performance_agg.drop(columns=['Fold'])
        df_metrics_performance_agg_flat, df_metrics_performance_agg_report = fold_results_aggregation(df_metrics_performance_agg)
        df_metrics_performance_agg_flat.round(2).to_csv(os.path.join(METRICS_DIR, f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_metric_then_aggregate_flat.csv'), index=False)
        df_metrics_performance_agg_report.round(2).to_csv(os.path.join(METRICS_DIR, f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_metric_then_aggregate_report.csv'), index=False)

        df_number_of_points_agg = pd.concat(aggregated_results_number_of_points, ignore_index=True)
        df_number_of_points_agg.to_csv(os.path.join(METRICS_DIR, f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_ALL_metric_then_aggregate.csv'), index=False)
        df_number_of_points_agg = df_number_of_points_agg.drop(columns=['Fold'])
        df_number_of_points_agg_flat, df_number_of_points_report = fold_results_aggregation(df_number_of_points_agg)
        df_number_of_points_agg_flat.to_csv(os.path.join(METRICS_DIR, f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_metric_then_aggregate_flat.csv'), index=False)
        df_number_of_points_report.to_csv(os.path.join(METRICS_DIR, f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_metric_then_aggregate_report.csv'), index=False)

        df_out_of_range_agg = pd.concat(aggregated_results_out_of_range, ignore_index=True)
        df_out_of_range_agg.to_csv(os.path.join(METRICS_DIR, f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_ALL_metric_then_aggregate.csv'), index=False)
        df_out_of_range_agg = df_out_of_range_agg.drop(columns=['Fold'])
        df_out_of_range_agg_flat, df_out_of_range_report = fold_results_aggregation(df_out_of_range_agg)
        df_out_of_range_agg_flat.to_csv(os.path.join(METRICS_DIR, f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_metric_then_aggregate_flat.csv'), index=False)
        df_out_of_range_report.to_csv(os.path.join(METRICS_DIR, f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_metric_then_aggregate_report.csv'), index=False)

        print('Aggregated metrics across folds (Metric-Then-Aggregate) calculated and saved.')
        print()

        # Save aggregated metrics across folds
        # df_all_folds_performance = pd.concat(aggregated_results_performance, ignore_index=True)
        # df_all_folds_performance.round(2).to_csv(os.path.join(METRICS_DIR, f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_ALL_metric_then_aggregate.csv'), index=False)
        #
        # df_all_folds_number_of_points = pd.concat(aggregated_results_number_of_points, ignore_index=True)
        # df_all_folds_number_of_points.to_csv(os.path.join(METRICS_DIR, f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_ALL_metric_then_aggregate.csv'), index=False)
        #
        # df_all_folds_out_of_range = pd.concat(aggregated_results_out_of_range, ignore_index=True)
        # df_all_folds_out_of_range.to_csv(os.path.join(METRICS_DIR, f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_ALL_metric_then_aggregate.csv'), index=False)



        #------------------ Aggregate - Then - Metric approach ------------------

        print()
        print('Calculating aggregated metrics across folds (Aggregate-Then-Metric)...')

        df_all_folds = pd.concat(aggregated_results, ignore_index=True)
        df_all_folds.to_parquet(os.path.join(PRED_DIR, f'df_test_results_vectors_{loss_function_name}_{algorithm}_H{horizon}.parquet'))
        print(f'\nAggregated all {k_folds} folds. Total rows: {len(df_all_folds)}')

        df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(df_all_folds, f'_{loss_function_name}_{algorithm}_H{horizon}_ALL', show_plot=True, plot_dir=FIGURES_DIR, minimum_sensor_reading=minimum_sensor_reading, maximum_sensor_reading=maximum_sensor_reading)
        df_metrics_performance.round(2).to_csv(os.path.join(METRICS_DIR, f'metrics_performance_by_range_{loss_function_name}_{algorithm}_H{horizon}_ALL_agregate_then_metric.csv'), index=False) # Change name agregate_then_metric
        df_number_of_points.to_csv(os.path.join(METRICS_DIR, f'number_of_points_by_zones_{loss_function_name}_{algorithm}_H{horizon}_ALL_agregate_then_metric.csv'), index=False)
        df_out_of_limits.to_csv(os.path.join(METRICS_DIR, f'predictions_out_of_limits_{loss_function_name}_{algorithm}_H{horizon}_ALL_agregate_then_metric.csv'), index=False)
        print('Metrics for all folds calculated and saved.')



        write_experiment_readme(EXP_DIR, loss_function_name, algorithm, horizon, history_length, experiment_context, k_folds)
        print(f"Main README.md created at: {os.path.join(EXP_DIR, 'README.md')}")
        
        print(f"================\nFinished Experiment: {os.path.basename(EXP_DIR)}\n================\n\n")

    finally:
        if exp_log_file:
            exp_log_file.close()
        sys.stdout = sys.stderr = Tee(original_stdout)

finally:
    print("All experiments finished.")
    sys.stdout = original_stdout
    sys.stderr = original_stderr