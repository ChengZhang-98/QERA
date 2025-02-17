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

plot_palette("cbf")

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


def plot_rxx_abs(rxx: np.array, ax, first_n_dims=64, cbar=False):
    # plot a histogram of the first_n_dims x first_n_dims elements of rxx
    rxx = np.abs(rxx[:first_n_dims, :first_n_dims])
    # normalize the rxx
    rxx = rxx / np.linalg.norm(rxx, ord="fro")
    vmin = np.quantile(rxx, 0.01)
    vmax = np.quantile(rxx, 0.99)
    sns.heatmap(
        rxx,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
        square=True,
        cbar_kws={"format": "%.0e"},
    )
    # hide the x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_model_rxx(model_rxx_dict, first_n_dims=64, cbar=False):
    num_heatmaps = len(model_rxx_dict)
    num_cols = 4
    num_rows = (num_heatmaps + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(1.5 * linewidth, 1.5 * linewidth * num_rows / num_cols),
    )

    for i, (label, rxx) in enumerate(model_rxx_dict.items()):
        row = i // num_cols
        col = i % num_cols
        if i < num_heatmaps:
            rxx = rxx.cpu().numpy().astype(np.float32)
            ax = axes[row, col]
            plot_rxx_abs(rxx, ax, first_n_dims=first_n_dims, cbar=cbar)
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
tinyllama_rxx_path = "scales_Cheng98_TinyLlama_v1.1_20240831-211748.safetensors"

tinyllama_rxx_dict = load_file(tinyllama_rxx_path)
# sort the dictionary by the layer number
tinyllama_rxx_dict = dict(
    sorted(
        tinyllama_rxx_dict.items(),
        key=lambda x: int(re.search(r"model\.layers\.(\d+)", x[0]).group(1)),
    )
)
# key
tinyllama_rxx_dict_k_proj = {
    k: v for k, v in tinyllama_rxx_dict.items() if "k_proj" in k
}
# o_proj
tinyllama_rxx_dict_o_proj = {
    k: v for k, v in tinyllama_rxx_dict.items() if "o_proj" in k
}
# down_proj
tinyllama_rxx_dict_down_proj = {
    k: v for k, v in tinyllama_rxx_dict.items() if "down_proj" in k
}
# gate_proj
tinyllama_rxx_dict_gate_proj = {
    k: v for k, v in tinyllama_rxx_dict.items() if "gate_proj" in k
}

fig_tinyllama = plot_model_rxx(tinyllama_rxx_dict_k_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_rxx_k_proj.pdf", bbox_inches="tight")

fig_tinyllama = plot_model_rxx(tinyllama_rxx_dict_o_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_rxx_o_proj.pdf", bbox_inches="tight")

fig_tinyllama = plot_model_rxx(tinyllama_rxx_dict_down_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_rxx_down_proj.pdf", bbox_inches="tight")

fig_tinyllama = plot_model_rxx(tinyllama_rxx_dict_gate_proj, first_n_dims=64)
fig_tinyllama.savefig("tinyllama_rxx_gate_proj.pdf", bbox_inches="tight")

# %%
llama_3_8b_rxx_path = "scales_meta-llama_Meta-Llama-3-8B.safetensors"
llama_3_8b_rxx_dict = load_file(llama_3_8b_rxx_path)
first_n_dims = 96
# sort the dictionary by the layer number
llama_3_8b_rxx_dict = dict(
    sorted(
        llama_3_8b_rxx_dict.items(),
        key=lambda x: int(re.search(r"model\.layers\.(\d+)", x[0]).group(1)),
    )
)

# key
llama_3_8b_rxx_dict_k_proj = {
    k: v for k, v in llama_3_8b_rxx_dict.items() if "k_proj" in k
}
# o_proj
llama_3_8b_rxx_dict_o_proj = {
    k: v for k, v in llama_3_8b_rxx_dict.items() if "o_proj" in k
}
# down_proj
llama_3_8b_rxx_dict_down_proj = {
    k: v for k, v in llama_3_8b_rxx_dict.items() if "down_proj" in k
}
# gate_proj
llama_3_8b_rxx_dict_gate_proj = {
    k: v for k, v in llama_3_8b_rxx_dict.items() if "gate_proj" in k
}

fig_llama_3_8b = plot_model_rxx(llama_3_8b_rxx_dict_k_proj, first_n_dims=first_n_dims)
fig_llama_3_8b.savefig("llama_3_8b_rxx_k_proj.pdf", bbox_inches="tight")

fig_llama_3_8b = plot_model_rxx(llama_3_8b_rxx_dict_o_proj, first_n_dims=first_n_dims)
fig_llama_3_8b.savefig("llama_3_8b_rxx_o_proj.pdf", bbox_inches="tight")

fig_llama_3_8b = plot_model_rxx(
    llama_3_8b_rxx_dict_down_proj, first_n_dims=first_n_dims
)
fig_llama_3_8b.savefig("llama_3_8b_rxx_down_proj.pdf", bbox_inches="tight")

fig_llama_3_8b = plot_model_rxx(
    llama_3_8b_rxx_dict_gate_proj, first_n_dims=first_n_dims
)
fig_llama_3_8b.savefig("llama_3_8b_rxx_gate_proj.pdf", bbox_inches="tight")

# %%
# 3.o_proj
fig, ax = plt.subplots(figsize=figsize)
plot_rxx_abs(
    llama_3_8b_rxx_dict["model.layers.3.self_attn.o_proj"],
    ax,
    first_n_dims=first_n_dims,
)
fig.savefig("llama_3_8b_rxx_layer_3_o_proj.pdf", bbox_inches="tight")

# 7.o_proj
fig, ax = plt.subplots(figsize=figsize)
plot_rxx_abs(
    llama_3_8b_rxx_dict["model.layers.7.self_attn.o_proj"],
    ax,
    first_n_dims=first_n_dims,
)
fig.savefig("llama_3_8b_rxx_layer_7_o_proj.pdf", bbox_inches="tight")

# 7.k_proj
fig, ax = plt.subplots(figsize=figsize)
plot_rxx_abs(
    llama_3_8b_rxx_dict["model.layers.7.self_attn.k_proj"],
    ax,
    first_n_dims=first_n_dims,
)
fig.savefig("llama_3_8b_rxx_layer_7_k_proj.pdf", bbox_inches="tight")

# 7.gate_proj
fig, ax = plt.subplots(figsize=figsize)
plot_rxx_abs(
    llama_3_8b_rxx_dict["model.layers.7.mlp.gate_proj"], ax, first_n_dims=first_n_dims
)
fig.savefig("llama_3_8b_rxx_layer_7_gate_proj.pdf", bbox_inches="tight")

# 7.down_proj
fig, ax = plt.subplots(figsize=figsize)
plot_rxx_abs(
    llama_3_8b_rxx_dict["model.layers.7.mlp.down_proj"], ax, first_n_dims=first_n_dims
)
fig.savefig("llama_3_8b_rxx_layer_7_down_proj.pdf", bbox_inches="tight")


# %%
llama_2_7b_rxx_path = "scales_meta-llama_Llama-2-7b-hf.safetensors"
llama_2_7b_rxx_dict = load_file(llama_2_7b_rxx_path)
first_n_dims = 96
# sort the dictionary by the layer number
llama_2_7b_rxx_dict = dict(
    sorted(
        llama_2_7b_rxx_dict.items(),
        key=lambda x: int(re.search(r"model\.layers\.(\d+)", x[0]).group(1)),
    )
)

# key
llama_2_7b_rxx_dict_k_proj = {
    k: v for k, v in llama_2_7b_rxx_dict.items() if "k_proj" in k
}
# o_proj
llama_2_7b_rxx_dict_o_proj = {
    k: v for k, v in llama_2_7b_rxx_dict.items() if "o_proj" in k
}
# down_proj
llama_2_7b_rxx_dict_down_proj = {
    k: v for k, v in llama_2_7b_rxx_dict.items() if "down_proj" in k
}
# gate_proj
llama_2_7b_rxx_dict_gate_proj = {
    k: v for k, v in llama_2_7b_rxx_dict.items() if "gate_proj" in k
}

fig_llama_2_7b = plot_model_rxx(llama_2_7b_rxx_dict_k_proj, first_n_dims=first_n_dims)
fig_llama_2_7b.savefig("llama_2_7b_rxx_k_proj.pdf", bbox_inches="tight")

fig_llama_2_7b = plot_model_rxx(llama_2_7b_rxx_dict_o_proj, first_n_dims=first_n_dims)
fig_llama_2_7b.savefig("llama_2_7b_rxx_o_proj.pdf", bbox_inches="tight")

fig_llama_2_7b = plot_model_rxx(
    llama_2_7b_rxx_dict_down_proj, first_n_dims=first_n_dims
)
fig_llama_2_7b.savefig("llama_2_7b_rxx_down_proj.pdf", bbox_inches="tight")

fig_llama_2_7b = plot_model_rxx(
    llama_2_7b_rxx_dict_gate_proj, first_n_dims=first_n_dims
)
fig_llama_2_7b.savefig("llama_2_7b_rxx_gate_proj.pdf", bbox_inches="tight")
