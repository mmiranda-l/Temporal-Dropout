import matplotlib.pyplot as plt
import numpy as np


# def plot_conf_matrix(ax, cf_matrix, add_title=""):
# 	ax = sns.heatmap(cf_matrix/np.sum(cf_matrix,axis=1, keepdims=True),
# 		annot=True, fmt='.2%', cmap='Blues',  vmin=0, vmax=1, cbar=False)
# 	ax.set_xlabel('\nPredicted Values')
# 	ax.set_ylabel('Actual Values ')
# 	ax.set_title(f"Confusion {add_title}")

def plot_dist_bin(ax, y_pred_cont, y_true, add_title=""):
    binwidth = 5 if np.max(y_true) > 50  else 0.5
    bins = np.arange(0, np.max(y_true)+ binwidth, binwidth) 
    ax.hist(y_true, label="Ground Truth", alpha=0.6, bins=bins , edgecolor='black', linewidth=1.2)
    ax.hist(y_pred_cont, label="Prediction", alpha=0.35, bins=bins,edgecolor='black', linewidth=1.2 )
    ax.set_title(f"Histogram of target values {add_title}")
    ax.legend(loc="upper right")
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    ax.set_xlim(0)

def plot_true_vs_pred(ax, y_pred_cont, y_true, fig, add_title=""):
    y = np.arange(np.min(y_true), np.max(y_true))
    ax.plot(y, y, "-", color="red")
    #ax.scatter(y_true, y_pred_cont, marker="o", edgecolors='black', s=30, rasterized=True)
    hb = plt.hexbin(y_true, y_pred_cont, gridsize=50, cmap='viridis', bins='log',)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Density')
    ax.set_title(f"Prediction vs ground truth {add_title}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")


def plot_true_vs_pred_uncertainty(ax, y_pred_cont, y_true, logvar, fig, add_title=""):
    sigma2 = np.sqrt(np.exp(logvar))
    epistemic = np.std(y_pred_cont, axis=-1)
    aleatoric = np.mean(sigma2, axis=-1)
    uncertainty = (epistemic + aleatoric) * 10
    y_pred_cont = y_pred_cont.mean(axis=-1)
    print(y_true.shape, y_pred_cont.shape, uncertainty)

    y = np.arange(np.min(y_true), np.max(y_true))
    ax.plot(y, y, "-", color="red")
    ax.scatter(y_true, y_pred_cont, marker="o", edgecolors='black', s=uncertainty, rasterized=True)
    ax.set_title(f"Prediction vs ground truth {add_title}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")

