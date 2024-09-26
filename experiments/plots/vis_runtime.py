# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

from loqer_exp.styles import set_default_style, get_cz_color, get_ic_color, plot_palette, cm2inch, get_cbf_color

# plot_palette("cz")
# plot_palette("ic")
plot_palette("cbf")

# %%
set_default_style()
linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth, linewidth * 0.75)
markersize = 4
FONT_SIZE_ANNO = 8
FONT_SIZE_S = 10
FONT_SIZE_M = 12
FONT_SIZE_L = 12
textweight = 600

plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M

# from accelerate import init_empty_weights
# from transformers import AutoModelForCausalLM
#
# model_names = [
# "Cheng98/TinyLlama_v1.1",
# "google/gemma-2-2b",
# "meta-llama/Llama-2-7b-hf",
# "meta-llama/Llama-2-13b-hf",
# "huggyllama/llama-30b",
# "meta-llama/Llama-2-70b-hf",
# ]
# num_params = []
# with init_empty_weights():
# for model_name in model_names:
# model = AutoModelForCausalLM.from_pretrained(model_name)
# num_params.append(model.num_parameters())

BAR_WIDTH = 1
# num_params = [1100048384, 3204165888, 6738415616, 13015864320]
num_params = [1.1, 2.6, 6.74, 13]
diag_calibration_times = [122.07024502754211, 238.45239925384521, 495.30890417099, 885.988844871521]
diag_sqrtm_times = [0, 0, 0, 0]
diag_svd_times = [114.09523868560791, 287.5154161453247, 1197.926025390625, 2453.2716178894043]
rxx_calibration_times = [652.5609645843506, 1458.0771751403809, 3159.7539417743683, 7190.371248722076]
rxx_sqrtm_times = [1602.4746532440186, 5858.455852985382, 12786.540725708008, 40417.62162208557]
rxx_svd_times = [82.32134294509888, 239.7993712425232, 943.7165603637695, 2182.253122329712]

# omit 2.6B
num_params = [1.1, 6.74, 13]
x_tick_labels = ["1B", "7B", "13B"]
diag_calibration_times = [122.07024502754211, 495.30890417099, 885.988844871521]
diag_sqrtm_times = [0, 0, 0]
diag_svd_times = [114.09523868560791, 1197.926025390625, 2453.2716178894043]
rxx_calibration_times = [652.5609645843506, 3159.7539417743683, 7190.371248722076]
rxx_sqrtm_times = [1602.4746532440186, 12786.540725708008, 40417.62162208557]
rxx_svd_times = [82.32134294509888, 943.7165603637695, 2182.253122329712]

num_params = np.array(num_params)
diag_calibration_times = np.array(diag_calibration_times)
diag_sqrtm_times = np.array(diag_sqrtm_times)
diag_svd_times = np.array(diag_svd_times)

rxx_calibration_times = np.array(rxx_calibration_times)
rxx_sqrtm_times = np.array(rxx_sqrtm_times)
rxx_svd_times = np.array(rxx_svd_times)

fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=1)

bottoms = np.zeros(len(num_params))
axes.bar(
    num_params,
    diag_calibration_times,
    BAR_WIDTH,
    # label="Calibration",
    color=get_cbf_color("cbf_lightblue"),
    bottom=bottoms,
)

bottoms += diag_calibration_times
axes.bar(
    num_params,
    diag_sqrtm_times,
    BAR_WIDTH,
    # label="Matrix sqrt",
    color=get_cbf_color("cbf_lightred"),
    bottom=bottoms,
)
bottoms += diag_sqrtm_times
axes.bar(
    num_params,
    diag_svd_times,
    BAR_WIDTH,
    # label="SVD",
    color=get_cbf_color("cbf_lightpurple"),
    bottom=bottoms,
)
total = diag_calibration_times + diag_sqrtm_times + diag_svd_times


def smart_time_format(t: float) -> str:
    # seconds to hours and minutes
    # return x.y hours, x.y minutes or x seconds

    if t < 60:
        return f"{t:.0f}s"
    elif t < 3600:
        return f"{t / 60:.1f}m"
    else:
        return f"{t / 3600:.1f}h"


# annotate on top of the bars
for i, (p, t) in enumerate(zip(num_params, total)):
    axes.annotate(
        f"{smart_time_format(t)}",
        (p, t),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=FONT_SIZE_ANNO,
        weight=textweight,
    )
    # annotate "A" under the bars
    axes.annotate(
        "A",
        (p, 0),
        textcoords="offset points",
        xytext=(0, -5),
        ha="center",
        fontsize=FONT_SIZE_ANNO,
        weight=textweight,
    )

# # rxx bar next to diag
bottoms = np.zeros(len(num_params))
axes.bar(
    num_params + BAR_WIDTH,
    rxx_calibration_times,
    BAR_WIDTH,
    label="Calibration",
    color=get_cbf_color("cbf_lightblue"),
    bottom=bottoms,
)

bottoms += rxx_calibration_times
axes.bar(
    num_params + BAR_WIDTH,
    rxx_sqrtm_times,
    BAR_WIDTH,
    label="Matrix sqrt",
    color=get_cbf_color("cbf_lightred"),
    bottom=bottoms,
)

bottoms += rxx_sqrtm_times
axes.bar(
    num_params + BAR_WIDTH,
    rxx_svd_times,
    BAR_WIDTH,
    label="SVD",
    color=get_cbf_color("cbf_lightpurple"),
    bottom=bottoms,
)
total = rxx_calibration_times + rxx_sqrtm_times + rxx_svd_times
# annotate on top of the bars
for i, (p, t) in enumerate(zip(num_params, total)):
    if i == 0:
        axes.annotate(
            f"{smart_time_format(t)}",
            (p + BAR_WIDTH, t * 1.6),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=FONT_SIZE_ANNO,
            weight=textweight,
        )
    else:
        axes.annotate(
            f"{smart_time_format(t)}",
            (p + BAR_WIDTH, t),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=FONT_SIZE_ANNO,
            weight=textweight,
        )
    # annotate "E" under the bars

    axes.annotate(
        "E",
        (p + BAR_WIDTH, 0),
        textcoords="offset points",
        xytext=(0, -5),
        ha="center",
        fontsize=FONT_SIZE_ANNO,
        weight=textweight,
    )


axes.set_xticks(num_params)
axes.set_xticklabels(x_tick_labels)
axes.set_xlim(0, 16)
axes.set_ylim(0, 6e4)
axes.set_yticks([0, 6 * 3600, 12 * 3600])
axes.set_yticklabels(["0", "6h", "12h"])
axes.set_xlabel("Model size")
axes.set_ylabel("Quantization time")
# put legend on top of axes[0]
axes.legend(loc="upper left")
plt.tight_layout()

# save the figure
fig.savefig("vis_runtime.pdf", bbox_inches="tight")

# %%
