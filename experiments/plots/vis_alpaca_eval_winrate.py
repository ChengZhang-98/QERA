# %%
import sys
from pathlib import Path
import re

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
from safetensors.torch import load_file

from loqer_exp.styles import set_default_style, get_cz_color, get_ic_color, plot_palette, cm2inch, get_color

plot_palette("cbf")
plot_palette("cz")

# %%
set_default_style()
linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth * 0.5, linewidth * 0.5 * 0.6)
markersize = 4
FONT_SIZE_S = 8
FONT_SIZE_M = 9
FONT_SIZE_L = 10

plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M

# vicuna-7b

# win, draw
vicuna_7b_identity_vs_w_only = [52.519789700215426, 100 * 25 / 804]
vicuna_7b_lqer_vs_w_only = [54.26312143298063, 100 * 20 / 805]
vicuna_7b_qera_exact_vs_w_only = [56.22011812159317, 100 * 24 / 805]


def plot_winrate(ax, rate_list, labels, xlim):
    """
    horizontal bar plot of win, draw, lose
    """

    width = 0.05

    y = 0
    for rates, label in zip(rate_list, labels):
        # win rate
        win = rates[0]
        draw = rates[1]
        lose = 100 - win - draw

        # plot stacked bar
        ax.barh(y, win, width, color=get_color("cbf_green"), alpha=0.8)
        ax.barh(y, draw, width, left=win, color=get_color("cbf_lightblue"), alpha=0.8)
        ax.barh(y, lose, width, left=win + draw, color=get_color("cbf_red"), alpha=0.8)

        # add text
        ax.text(win / 2, y, f"{win:.1f}%", ha="center", va="center", color="#071810")
        # ax.text(win + draw / 2, y, f"{draw:.1f}%", ha="center", va="center", color="black")
        ax.text(win + draw + lose / 2, y, f"{lose:.1f}%", ha="center", va="center", color="#1F0808")

        # annotate y label
        ax.text(xlim[0] - 5, y, label, ha="right", va="center", color="black")

        y -= 1.5 * width

    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["bottom", "top"]].set_visible(False)

    # put the legend outside
    ax.legend(["Win", "Tie", "Lost"], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)


fig, ax = plt.subplots(1, 1, figsize=figsize)
plot_winrate(
    ax,
    [vicuna_7b_identity_vs_w_only, vicuna_7b_lqer_vs_w_only, vicuna_7b_qera_exact_vs_w_only],
    ["ZeroQ", "LQER", "QERA"],
    xlim=[0, 100],
)
# set y label
plt.tight_layout()
plt.show()

# %%
# save the figure

fig.savefig("vicuna_alpaca_eval.pdf", bbox_inches="tight")
