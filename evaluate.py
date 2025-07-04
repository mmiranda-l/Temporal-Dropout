import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rc('font', **{"size":14})

from src.datasets.views_structure import load_structure
from src.metrics.metrics import RegressionMetrics
from src.visualizations.utils import save_results, gt_mask
from src.visualizations.tools import plot_dist_bin,plot_true_vs_pred

def load_data_sup(data_name, method_name, dir_folder="", **args):
    files_load = [str(v) for v in Path(f"{dir_folder}/pred/{data_name}/{method_name}").glob(f"*.csv")]
    files_load.sort()

    preds_p_run = []
    indxs_p_run = []
    for file_n in files_load:
        data_views = pd.read_csv(file_n, index_col=0) #load_structure(file_n)
        preds_p_run.append(data_views.values)
        indxs_p_run.append(list(data_views.index))
    return preds_p_run,indxs_p_run

def evaluate(
                preds_p_run,
                indexs_p_run,
                data_ground_truth,
                ind_save,
                plot_runs = False,
                include_metrics = [],
                dir_folder = "",
                ):
    R = len(preds_p_run)

    df_runs = []
    df_runs_diss = []
    
    y_true_concatenated = [] #to create 
    y_pred_concatenated = []
    for r in range(R):
        y_true, y_pred = gt_mask(data_ground_truth, indexs_p_run[r]), preds_p_run[r]
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        y_true_concatenated.append(y_true)#to create plots

        if len(y_pred.shape)==2 and y_pred.shape[1] > 1: #calculate mean of predictions for evaluation
            y_pred_mean = y_pred.mean(axis=-1)
        else:
            y_pred_mean = y_pred
        y_pred_concatenated.append(y_pred_mean)#to create plots

        d_me = RegressionMetrics()
        dic_res = d_me(y_pred_mean, y_true)
        df_res = pd.DataFrame(dic_res, index=["test"])
        df_runs.append(df_res)

        #For MIRO: you can add here more evaluations based on y_pred (that will be (n_samples, n_predictions) )

    df_concat = pd.concat(df_runs).groupby(level=0)
    df_mean = df_concat.mean()
    df_std = df_concat.std()

    print(df_concat)
    all_df = pd.concat(df_runs)
    all_df.index = [f"fold-{v:02d}" for v in range(R)]
    save_results(f"{dir_folder}/plots/{ind_save}/results_all", all_df) #store per group
    save_results(f"{dir_folder}/plots/{ind_save}/preds_mean", df_mean)
    save_results(f"{dir_folder}/plots/{ind_save}/preds_std", df_std)
    print(f"################ Showing the {ind_save} ################")
    print(df_mean.round(4).to_markdown())
    print(df_std.round(4).to_markdown())

    #overall plots (across al folds)
    y_pred_concatenated = np.concatenate(y_pred_concatenated,axis=0)
    y_true_concatenated = np.concatenate(y_true_concatenated,axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
    plot_dist_bin(ax[0,0], y_pred_concatenated, y_true_concatenated)
    save_results(f"{dir_folder}/plots/{ind_save}/prediction_histogram", plt)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5), squeeze=False)
    plot_true_vs_pred(ax[0,0], y_pred_concatenated, y_true_concatenated)
    save_results(f"{dir_folder}/plots/{ind_save}/predictions_vs_groundtruth", plt)
    plt.close("all")
    plt.clf()

    return df_mean,df_std

def calculate_metrics(df_summary, df_std, data_te,data_name, method, **args):
    preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method, **args)

    df_aux, df_aux2= evaluate(
                        preds_p_run_te,
                        indexs_p_run_te,
                        data_te,
                        ind_save=f"{data_name}/{method}/",
                        **args
                        )
    df_summary[method] = df_aux.loc["test"]
    df_std[method] = df_aux2.loc["test"]

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    include_metrics = ["f1 bin", "p bin"]

    data_tr = load_structure(f"{input_dir_folder}/{data_name}.nc")

    methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/test"))

    df_summary_sup, df_summary_sup_s = pd.DataFrame(), pd.DataFrame()
    pool_names = {}
    missview_methods = {}
    for method in methods_to_plot:
        print(f"Evaluating method {method}")
        calculate_metrics(df_summary_sup, df_summary_sup_s,
                        data_tr, 
                        data_name,
                        method,
                        include_metrics=include_metrics,
                        plot_runs=config_file.get("plot_runs"),
                        dir_folder=output_dir_folder,
                        )

    #all figures were saved in output_dir_folder/plots
    print(">>>>>>>>>>>>>>>>> Mean across runs on test set")
    print((df_summary_sup.T).round(4).to_markdown())
    print(">>>>>>>>>>>>>>>>> Std across runs on test set")
    print((df_summary_sup_s.T).round(4).to_markdown())
    df_summary_sup.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_mean.csv")
    df_summary_sup_s.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_std.csv")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_evaluation(config_file)
