# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import numpy as np
import matplotlib.pyplot as plt
import yaml

from qera_exp.styles import set_default_style, plot_palette

plot_palette("cbf")

# %%

with open("./roberta_weight_approx_error_3bit.yaml", "r") as f:
    approx_error = yaml.safe_load(f)

loftq_layer_to_errors_raw: dict[str, list[float]] = approx_error[
    "Cheng98_roberta-base_rank_16_bit_3_adapter_loftq_loftq-5-iter"
]
qera_layer_to_error_raw: dict[str, list[float]] = approx_error[
    "Cheng98_roberta-base_rank_16_bit_3_adapter_qera_loftq-0-iter"
]

layer_name_patterns = [
    "self.key",
    "self.query",
    "self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]


def extract_layer_num(layer_name: str) -> int:
    return int(layer_name.split(".")[5])


loftq_errors_dict = {}
qera_error_dict = {}

for layer_name in loftq_layer_to_errors_raw:
    for layer_name_p in layer_name_patterns:
        if layer_name_p in layer_name:
            if layer_name_p not in loftq_errors_dict:
                loftq_errors_dict[layer_name_p] = {}
                qera_error_dict[layer_name_p] = {}

            layer_number = extract_layer_num(layer_name)
            loftq_errors_dict[layer_name_p][layer_number] = loftq_layer_to_errors_raw[
                layer_name
            ]
            qera_error_dict[layer_name_p][layer_number] = qera_layer_to_error_raw[
                layer_name.removeprefix("base_model.model.")
            ]
            break

    # sort by layer number
    loftq_errors_dict[layer_name_p] = dict(
        sorted(loftq_errors_dict[layer_name_p].items())
    )
    qera_error_dict[layer_name_p] = dict(sorted(qera_error_dict[layer_name_p].items()))


# %%

linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth * 0.8, linewidth * 0.8 * 0.75)
markersize = 6
FONT_SIZE_S = 7
FONT_SIZE_M = 8
FONT_SIZE_L = 10

set_default_style()
plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M


def plot_error_vs_num_iters(
    loftq_errors_list: list[list[float]], loftq_labels: list[str], ax, color_map
):

    assert len(loftq_errors_list) == len(loftq_labels)
    assert all(
        len(loftq_errors) == len(loftq_errors_list[0])
        for loftq_errors in loftq_errors_list
    )

    num_iters = len(loftq_errors_list[0])
    x = list(range(1, num_iters + 1))

    colors = color_map(np.linspace(0.1, 0.9, len(loftq_errors_list)))

    for i, (loftq_errors, label, color) in enumerate(
        zip(loftq_errors_list, loftq_labels, colors)
    ):
        ax.plot(x, loftq_errors, "^-", markersize=markersize, color=color, label=label)

    ax.set_xlabel(r"LoftQ num iterations")
    ax.set_xticks(x)

    return ax


# gist_heat
color_map = plt.get_cmap("gist_heat")
title_map = {
    "self.key": "key_proj",
    "self.query": "query_proj",
    "self.value": "value_proj",
    "attention.output.dense": "out_proj",
    "intermediate.dense": "fc1",
    "output.dense": "fc2",
}

# 3: 2
fig, axs = plt.subplots(2, 3, figsize=(1.2 * linewidth, 1.2 * linewidth / 3 * 2))

for i, (layer_name_p, loftq_errors) in enumerate(loftq_errors_dict.items()):
    loftq_errors_list = list(loftq_errors.values())
    loftq_labels = list(loftq_errors.keys())
    loftq_labels_layer_number = [f"Layer {i}" for i in loftq_labels]

    ax = axs[i // 3, i % 3]
    plot_error_vs_num_iters(loftq_errors_list, loftq_labels_layer_number, ax, color_map)
    ax.set_title(title_map[layer_name_p])

    if i == 0 or i == 3:
        ax.set_ylabel(r"Weight reconstruction error")

    # if i == 2:
    #     # add legend, upper right, outside the plot
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    #     # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=4)
    if i == 5:
        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=6)
fig.tight_layout()

fig.savefig("roberta_weight_approx_error_vs_num_iters_3bit.pdf", bbox_inches="tight")

# %%
