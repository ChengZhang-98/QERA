# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import matplotlib.pyplot as plt

from loqer_exp.styles import set_default_style, get_color


# %%
set_default_style()
linewidth = 5.5  # inch
lineheight = 9  # inch
figsize = (linewidth * 0.5, linewidth * 0.5 * 0.75)
markersize = 4
FONT_SIZE_S = 8
FONT_SIZE_M = 10
FONT_SIZE_L = 11

plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M

# num calibration samples
ppl_identity = 13.9090
num_samples = [4, 8, 16, 64, 256]
ppl_lqer = [10.7185, 10.6929, 10.7392, 10.7247, 10.7833]
ppl_loqer_diag = [10.7007, 10.6807, 10.6749, 10.6586, 10.6594]

fig, ax = plt.subplots(figsize=figsize)

ax.plot(num_samples, ppl_lqer, "s-", label="LQER", markersize=markersize, color=get_color("cbf_purple"))
ax.plot(num_samples, ppl_loqer_diag, "o-", label="QERA-approx", markersize=markersize, color=get_color("cbf_green"))
# ax.axhline(y=ppl_identity, color=get_cz_color("cz_lightorange"), linestyle="--", label="Identity")

# log2 x-axis
ax.set_xscale("log", base=2)
ax.set_xticks(num_samples)
ax.set_xlabel("Num calibration samples")
ax.set_ylabel("Perplexity ($\downarrow$)")
ax.legend()

fig.savefig("tinyllama_ppl_vs_num_calibration_samples.pdf", bbox_inches="tight")

# %%
