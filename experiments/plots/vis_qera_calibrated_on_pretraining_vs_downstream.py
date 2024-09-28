# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import pandas as pd
import matplotlib.pyplot as plt

from loqer_exp.styles import set_default_style, plot_palette, get_color

plot_palette("cbf")
# %%

linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth, linewidth * 0.5)
markersize = 0.25
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

df = pd.read_csv("./roberta-qera-calibrated-on-pretraining-vs-downstream.csv")
# calibrated on pretrianing dataset
loss_lr1_pre = df.loc[:, "Pretraining_lr-1e-4 - train_step_loss"]
loss_lr2_pre = df.loc[:, "Pretraining_lr-2e-4 - train_step_loss"]
loss_lr3_pre = df.loc[:, "Pretraining_lr-3e-4 - train_step_loss"]
# calibrated on downstream dataset
loss_lr1_down = df.loc[:, "Downstream_lr-1e-4 - train_step_loss"]
loss_lr2_down = df.loc[:, "Downstream_lr-2e-4 - train_step_loss"]
loss_lr3_down = df.loc[:, "Downstream_lr-3e-4 - train_step_loss"]

window_size = 40

loss_lr1_pre = loss_lr1_pre.rolling(window=window_size).mean()
loss_lr2_pre = loss_lr2_pre.rolling(window=window_size).mean()
loss_lr3_pre = loss_lr3_pre.rolling(window=window_size).mean()

loss_lr1_down = loss_lr1_down.rolling(window=window_size).mean()
loss_lr2_down = loss_lr2_down.rolling(window=window_size).mean()
loss_lr3_down = loss_lr3_down.rolling(window=window_size).mean()

fig, ax = plt.subplots(1, 1, figsize=figsize)

ax.plot(loss_lr1_down, label="Downstream_lr-1e-4", color=get_color("cbf_purple"), linestyle="-")
ax.plot(loss_lr1_pre, label="Pretraining_lr-1e-4", color=get_color("cbf_darkgreen"), linestyle="-")
ax.plot(loss_lr2_down, label="Downstream_lr-2e-4", color=get_color("cbf_red"), linestyle="-")
ax.plot(loss_lr2_pre, label="Pretraining_lr-2e-4", color=get_color("cbf_green"), linestyle="-")
ax.plot(loss_lr3_down, label="Downstream_lr-3e-4", color=get_color("cbf_lightred"), linestyle="-")
ax.plot(loss_lr3_pre, label="Pretraining_lr-3e-4", color=get_color("cbf_lightgreen"), linestyle="-")

ax.set_xlabel("Step")
ax.set_ylabel("Loss")
# put legend outside of the plot
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)

plt.tight_layout()

# save to pdf
plt.savefig("roberta-qera-calibrated-on-pretraining-vs-downstream.pdf", bbox_inches="tight")
