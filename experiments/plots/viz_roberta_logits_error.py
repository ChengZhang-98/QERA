# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

from loqer_exp.styles import set_default_style, get_cz_color, get_ic_color, plot_palette, cm2inch

plot_palette("cz")
plot_palette("ic")

# %%


def visualize_roberta_logits_error(df):
    # Index(['rank', 'bits', 'adapter_init', 'output_error', 'quant_type'], dtype='object')

    # plotting arguments
    set_default_style()
    linewidth = 5.5  # inch
    lineheight = 9  # inch
    figsize = (linewidth * 0.8, linewidth * 0.8 * 0.75)
    markersize = 6
    FONT_SIZE_S = 10
    FONT_SIZE_M = 11
    FONT_SIZE_L = 12

    plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE_L)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
    plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
    plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
    plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M
    fig, ax = plt.subplots(figsize=figsize)

    # output_error-vs-rank
    # 4-bit, qlora
    mask = (df["bits"] == 4) & (df["adapter_init"] == "qlora")
    plt.plot(
        np.arange(len(df.loc[mask, "rank"])),
        df.loc[mask, "output_error"],
        "x--",
        markersize=markersize,
        color=get_cz_color("cz_grey"),
        label="QLoRA",
    )

    # 4-bit, loftq (1-iter)
    mask = (df["bits"] == 4) & (df["adapter_init"] == "loftq (1-iter)")
    plt.plot(
        np.arange(len(df.loc[mask, "rank"])),
        df.loc[mask, "output_error"],
        "^--",
        markersize=markersize,
        color=get_cz_color("cz_darkred"),
        label="LoftQ (1-iter)",
    )

    # 4-bit, loftq (3-iter)
    mask = (df["bits"] == 4) & (df["adapter_init"] == "loftq (3-iter)")
    plt.plot(
        np.arange(len(df.loc[mask, "rank"])),
        df.loc[mask, "output_error"],
        "^--",
        markersize=markersize,
        color=get_cz_color("cz_lightred"),
        label="LoftQ (3-iter)",
    )

    # 4-bit, loftq (5-iter)
    mask = (df["bits"] == 4) & (df["adapter_init"] == "loftq (5-iter)")
    plt.plot(
        np.arange(len(df.loc[mask, "rank"])),
        df.loc[mask, "output_error"],
        "^--",
        markersize=markersize,
        color=get_cz_color("cz_lightorange"),
        label="LoftQ (5-iter)",
    )

    # 4-bit, loqer
    mask = (df["bits"] == 4) & (df["adapter_init"] == "loqer")
    plt.plot(
        np.arange(len(df.loc[mask, "rank"])),
        df.loc[mask, "output_error"],
        "o--",
        markersize=markersize,
        color=get_cz_color("cz_green"),
        label="LoQER",
    )

    ax.set_ylabel(r"Output logits error")
    ax.set_xlabel(r"Rank")
    ax.set_xticks(np.arange(len(df.loc[mask, "rank"])))
    ax.set_xticklabels(df.loc[mask, "rank"].unique())
    # place the legend outside the plot, right side
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="4-bit Weight")
    plt.show()
    return fig, ax


df_4bit = pd.read_pickle("roberta_output_error_4bit.pkl")
df_4bit = df_4bit.sort_values(by=["adapter_init", "rank"])
print(df_4bit)
fig_output_e, ax_output_e = visualize_roberta_logits_error(df_4bit)

# %%
fig_output_e.savefig("roberta_output_error_vs_rank_4bit.pdf", bbox_inches="tight")


# %%


def visualize_roberta_loftq_output_errors_3bit(df):
    set_default_style()
    linewidth = 5.5  # inch
    lineheight = 9  # inch
    figsize = (linewidth * 0.8, linewidth * 0.8 * 0.75)
    markersize = 6
    FONT_SIZE_S = 10
    FONT_SIZE_M = 11
    FONT_SIZE_L = 12

    plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE_L)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
    plt.rc("legend", fontsize=FONT_SIZE_S)  # legend fontsize
    plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title
    plt.rcParams["legend.title_fontsize"] = FONT_SIZE_M
    fig, ax = plt.subplots(figsize=figsize)

    mask_loftq_r4 = (
        (df["adapter_init"].str.contains("loftq"))
        & (df["rank"] == 4)
        & (df["bits"] == 3)
        & (df["quant_type"] == "mxint")
    )
    mask_loftq_r8 = (
        (df["adapter_init"].str.contains("loftq"))
        & (df["rank"] == 8)
        & (df["bits"] == 3)
        & (df["quant_type"] == "mxint")
    )
    mask_loftq_r16 = (
        (df["adapter_init"].str.contains("loftq"))
        & (df["rank"] == 16)
        & (df["bits"] == 3)
        & (df["quant_type"] == "mxint")
    )
    mask_loqer_r4 = (
        (df["adapter_init"] == "loqer") & (df["rank"] == 4) & (df["bits"] == 3) & (df["quant_type"] == "mxint")
    )
    mask_loqer_r8 = (
        (df["adapter_init"] == "loqer") & (df["rank"] == 8) & (df["bits"] == 3) & (df["quant_type"] == "mxint")
    )
    mask_loqer_r16 = (
        (df["adapter_init"] == "loqer") & (df["rank"] == 16) & (df["bits"] == 3) & (df["quant_type"] == "mxint")
    )
    mask_qlora = (df["adapter_init"] == "qlora") & (df["rank"] == 8) & (df["bits"] == 3) & (df["quant_type"] == "mxint")

    # output error vs loftq num iters
    qlora_error = df.loc[mask_qlora, "output_error"].values[0]
    plt.axhline(qlora_error, color=get_cz_color("cz_grey"), linestyle="-.", label="QLoRA")

    num_iters = np.arange(len(df.loc[mask_loftq_r8, :])) + 1
    plt.plot(
        num_iters,
        df.loc[mask_loftq_r4, "output_error"],
        "^--",
        markersize=markersize,
        color=get_cz_color("cz_darkred"),
        label="LoftQ (k=4)",
    )
    plt.plot(
        num_iters,
        df.loc[mask_loftq_r8, "output_error"],
        "^:",
        markersize=markersize,
        color=get_cz_color("cz_lightred"),
        label="LoftQ (k=8)",
    )
    plt.plot(
        num_iters,
        df.loc[mask_loftq_r16, "output_error"],
        "^-",
        markersize=markersize,
        color=get_cz_color("cz_lightorange"),
        label="LoftQ (k=16)",
    )

    loqer_error_r4 = df.loc[mask_loqer_r4, "output_error"].values[0]
    plt.axhline(loqer_error_r4, color=get_cz_color("cz_darkgreen"), linestyle="--", label="LoQER (k=4)")
    loqer_error_r8 = df.loc[mask_loqer_r8, "output_error"].values[0]
    plt.axhline(loqer_error_r8, color=get_cz_color("cz_green"), linestyle=":", label="LoQER (k=8)")
    loqer_error_r16 = df.loc[mask_loqer_r16, "output_error"].values[0]
    plt.axhline(loqer_error_r16, color=get_cz_color("cz_lightgreen"), linestyle="-", label="LoQER (k=16)")

    ax.set_ylabel(r"Output logits error")
    ax.set_xlabel(r"LoftQ num iterations")
    ax.set_xticks(num_iters)

    # place the legend outside the plot, right side
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="3-bit Weight")
    # ax.legend()

    plt.show()

    return fig, ax


df_3bit = pd.read_pickle("roberta_output_error_3bit.pkl")
df_3bit = df_3bit.sort_values(by=["adapter_init", "rank"])
print(df_3bit)
fig, ax = visualize_roberta_loftq_output_errors_3bit(df_3bit)
fig.savefig("roberta_output_error_vs_loftq_iters_3bit.pdf", bbox_inches="tight")

# %%
