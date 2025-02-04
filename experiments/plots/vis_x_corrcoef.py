# %%
import sys
from pathlib import Path
import re

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file

from qera_exp.styles import set_default_style, plot_palette

# %%
set_default_style()
linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth * 0.5, linewidth * 0.5 * 0.75)
markersize = 4
FONT_SIZE_S = 6
FONT_SIZE_M = 6
FONT_SIZE_L = 6

plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M


def plot_corrcoef(coef: np.array, ax, first_n_dims=64, cbar=True):
    # plot a histogram of the first_n_dims x first_n_dims elements of coef
    coef = coef[:first_n_dims, :first_n_dims]
    sns.heatmap(
        coef,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        cbar=cbar,
        square=True,
        cbar_kws={"format": "%.2f"},
    )
    # hide the x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_model_corrcoef(model_coef_dict, first_n_dims=64):
    num_heatmaps = len(model_coef_dict)
    num_cols = 4
    num_rows = (num_heatmaps + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(1.5 * linewidth, 1.5 * linewidth * num_rows / num_cols),
    )

    for i, (label, coef) in enumerate(model_coef_dict.items()):
        row = i // num_cols
        col = i % num_cols
        if i < num_heatmaps:
            coef = coef.cpu().numpy().astype(np.float32)
            ax = axes[row, col]
            plot_corrcoef(coef, ax, first_n_dims=first_n_dims, cbar=True)
            layer_num = int(label.split(".")[2])
            layer_name = label.split(".")[-1]
            label = f"{layer_name} (Layer {layer_num})"

            ax.set_title(label)

    # hide the empty axes
    for i in range(num_heatmaps, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.axis("off")

    fig.tight_layout()
    return fig


# %%
# tinyllama
tinyllama_coef_path = "corrcoef_TinyLlama_v1.1_20240902-112933.safetensors"

tinyllama_coef_dict = load_file(tinyllama_coef_path)
# sort the dictionary by the layer number
tinyllama_coef_dict = dict(
    sorted(
        tinyllama_coef_dict.items(),
        key=lambda x: int(re.search(r"model\.layers\.(\d+)", x[0]).group(1)),
    )
)
# key
tinyllama_coef_dict_k_proj = {
    k: v for k, v in tinyllama_coef_dict.items() if "k_proj" in k
}
# o_proj
tinyllama_coef_dict_o_proj = {
    k: v for k, v in tinyllama_coef_dict.items() if "o_proj" in k
}
# down_proj
tinyllama_coef_dict_down_proj = {
    k: v for k, v in tinyllama_coef_dict.items() if "down_proj" in k
}
# gate_proj
tinyllama_coef_dict_gate_proj = {
    k: v for k, v in tinyllama_coef_dict.items() if "gate_proj" in k
}

fig_tinyllama = plot_model_corrcoef(tinyllama_coef_dict_k_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_coef_proj.pdf", bbox_inches="tight")

fig_tinyllama = plot_model_corrcoef(tinyllama_coef_dict_o_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_coef_o_proj.pdf", bbox_inches="tight")

fig_tinyllama = plot_model_corrcoef(tinyllama_coef_dict_down_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_coef_down_proj.pdf", bbox_inches="tight")

fig_tinyllama = plot_model_corrcoef(tinyllama_coef_dict_gate_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_coef_gate_proj.pdf", bbox_inches="tight")

# %%
llama_3_8b_coef_path = "corrcoef_meta-llama_Meta-Llama-3-8B_20240902-114311.safetensors"
llama_3_8b_coef_dict = load_file(llama_3_8b_coef_path)
# sort the dictionary by the layer number
llama_3_8b_coef_dict = dict(
    sorted(
        llama_3_8b_coef_dict.items(),
        key=lambda x: int(re.search(r"model\.layers\.(\d+)", x[0]).group(1)),
    )
)

# key
llama_3_8b_coef_dict_k_proj = {
    k: v for k, v in llama_3_8b_coef_dict.items() if "k_proj" in k
}
# o_proj
llama_3_8b_coef_dict_o_proj = {
    k: v for k, v in llama_3_8b_coef_dict.items() if "o_proj" in k
}
# down_proj
llama_3_8b_coef_dict_down_proj = {
    k: v for k, v in llama_3_8b_coef_dict.items() if "down_proj" in k
}
# gate_proj
llama_3_8b_coef_dict_gate_proj = {
    k: v for k, v in llama_3_8b_coef_dict.items() if "gate_proj" in k
}

fig_llama_3_8b = plot_model_corrcoef(llama_3_8b_coef_dict_k_proj, first_n_dims=64)
fig_llama_3_8b.savefig("llama_3_8b_coef_k_proj.pdf", bbox_inches="tight")

fig_llama_3_8b = plot_model_corrcoef(llama_3_8b_coef_dict_o_proj, first_n_dims=64)
fig_llama_3_8b.savefig("llama_3_8b_coef_o_proj.pdf", bbox_inches="tight")

fig_llama_3_8b = plot_model_corrcoef(llama_3_8b_coef_dict_down_proj, first_n_dims=64)
fig_llama_3_8b.savefig("llama_3_8b_coef_down_proj.pdf", bbox_inches="tight")

fig_llama_3_8b = plot_model_corrcoef(llama_3_8b_coef_dict_gate_proj, first_n_dims=64)
fig_llama_3_8b.savefig("llama_3_8b_coef_gate_proj.pdf", bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=figsize)
plot_corrcoef(
    llama_3_8b_coef_dict["model.layers.0.self_attn.k_proj"],
    ax,
    first_n_dims=64,
    cbar=False,
)
fig.savefig("llama_3_8b_coef_layer_0_k_proj.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize)
plot_corrcoef(
    llama_3_8b_coef_dict["model.layers.3.self_attn.k_proj"],
    ax,
    first_n_dims=64,
    cbar=False,
)
fig.savefig("llama_3_8b_coef_layer_3_k_proj.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize)
plot_corrcoef(
    llama_3_8b_coef_dict["model.layers.3.self_attn.o_proj"],
    ax,
    first_n_dims=64,
    cbar=False,
)
fig.savefig("llama_3_8b_coef_layer_3_o_proj.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize)
plot_corrcoef(
    llama_3_8b_coef_dict["model.layers.3.mlp.down_proj"], ax, first_n_dims=64, cbar=True
)
fig.savefig("llama_3_8b_coef_layer_3_down_proj.pdf", bbox_inches="tight")

# %%
