# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

from loqer_exp.styles import set_default_style, get_cz_color, get_ic_color, plot_palette, get_color

plot_palette("cz")
plot_palette("ic")
plot_palette("cbf")

# %%

# spearman correlation
seed_42_random_stsb = -0.04984204584412876
stsb_loqer = [seed_42_random_stsb, 0.840622051, 0.858967945, 0.872314011, 0.874235205, 0.875514633]
stsb_loftq = [seed_42_random_stsb, 0.557813152, 0.818315977, 0.852196865, 0.848014452, 0.852296297]
stsb_qlora = [seed_42_random_stsb, 0.282983845, 0.809318388, 0.838049963, 0.841832246, 0.846087864]

# accuracy
seed_42_random_mrpc = 0.3161764705882353
mrpc_loqer = [seed_42_random_mrpc, 0.7009803921568627, 0.75, 0.7696078431372549, 0.7769607843137255, 0.7769607843137255]
mrpc_loftq = [
    seed_42_random_mrpc,
    0.6838235294117647,
    0.7083333333333334,
    0.7450980392156863,
    0.7524509803921569,
    0.7524509803921569,
]
mrpc_qlora = [
    seed_42_random_mrpc,
    0.6838235294117647,
    0.696078431,
    0.7303921568627451,
    0.7328431372549019,
    0.7426470588235294,
]

set_default_style()
linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth * 0.5, linewidth * 0.5 * 0.75)
markersize = 5
FONT_SIZE_S = 7
FONT_SIZE_M = 8
FONT_SIZE_L = 10

plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M

color_map = {
    "loqer": get_color("cbf_green"),
    "loftq": get_color("cbf_orange"),
    "qlora": get_color("cbf_grey"),
}

label_map = {
    "loqer": "LoQER (diag)",
    "loftq": "LoftQ (5-iter)",
    "qlora": "QLoRA",
}

marker_map = {
    "loqer": "o--",
    "loftq": "^--",
    "qlora": "x--",
}


def plot_metric_vs_epoch(metrics: list[float], epoch: list[int], ax, labels: list[str]):
    assert all(len(metric) == len(epoch) for metric in metrics)
    assert len(metrics[0]) == len(epoch)
    assert len(metrics) == len(labels)

    x = epoch
    y = metrics

    for i, (y_i, label) in enumerate(zip(y, labels)):
        ax.plot(x, y_i, marker_map[label], markersize=markersize, color=color_map[label], label=label_map[label])

    ax.set_xlabel("Epoch")
    return ax


fig_stsb, ax_stsb = plt.subplots(1, 1, figsize=(figsize))
plot_metric_vs_epoch(
    [stsb_qlora, stsb_loftq, stsb_loqer], list(range(len(stsb_loqer))), ax_stsb, ["qlora", "loftq", "loqer"]
)
ax_stsb.set_ylabel("Spearman Correlation")
ax_stsb.legend()
fig_stsb.savefig("roberta_stsb_convergence_3bit.pdf", bbox_inches="tight")

fig_mrpc, ax_mrpc = plt.subplots(1, 1, figsize=(figsize))
plot_metric_vs_epoch(
    [mrpc_qlora, mrpc_loftq, mrpc_loqer], list(range(len(mrpc_loqer))), ax_mrpc, ["qlora", "loftq", "loqer"]
)
ax_mrpc.set_ylabel("Accuracy")
ax_mrpc.legend()
fig_mrpc.savefig("roberta_mrpc_convergence_3bit.pdf", bbox_inches="tight")

# %%
