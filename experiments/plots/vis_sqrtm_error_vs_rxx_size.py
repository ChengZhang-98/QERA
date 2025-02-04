# %%
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
import pandas as pd
import matplotlib.pyplot as plt

from qera_exp.styles import set_default_style, plot_palette, get_color


def visualize_sqrtm_error_vs_rxx_size():
    raw_data = [
        # 0
        ["TinyLlama-1.1B", "model.layers.0.mlp.down_proj", 5632, 2.983556069852429e-22],
        ["Llama-2-7B", "model.layers.0.mlp.down_proj", 11008, 7.52263496894101e-24],
        ["Llama-2-13B", "model.layers.0.mlp.down_proj", 13824, 7.976008702989482e-25],
        ["Llama-2-70B", "model.layers.0.mlp.down_proj", 28672, 1.4446307227077917e-23],
        # 1
        ["TinyLlama-1.1B", "model.layers.1.mlp.down_proj", 5632, 1.629884787245808e-23],
        ["Llama-2-7B", "model.layers.1.mlp.down_proj", 11008, 1.0036187512794466e-20],
        ["Llama-2-13B", "model.layers.1.mlp.down_proj", 13824, 2.531538386945739e-23],
        ["Llama-2-70B", "model.layers.1.mlp.down_proj", 28672, 1.302859656789357e-22],
        # 4
        [
            "TinyLlama-1.1B",
            "model.layers.4.mlp.down_proj",
            5632,
            2.5193668002226372e-23,
        ],
        ["Llama-2-7B", "model.layers.4.mlp.down_proj", 11008, 3.3046589408859624e-22],
        ["Llama-2-13B", "model.layers.4.mlp.down_proj", 13824, 9.7150767403211e-22],
        ["Llama-2-70B", "model.layers.4.mlp.down_proj", 28672, 1.1931670913721443e-21],
        # 8
        ["TinyLlama-1.1B", "model.layers.8.mlp.down_proj", 5632, 7.413649389529984e-23],
        ["Llama-2-7B", "model.layers.8.mlp.down_proj", 11008, 1.829556955668731e-21],
        ["Llama-2-13B", "model.layers.8.mlp.down_proj", 13824, 2.5927970350730244e-21],
        ["Llama-2-70B", "model.layers.8.mlp.down_proj", 28672, 1.278229530646947e-19],
        # 12
        [
            "TinyLlama-1.1B",
            "model.layers.12.mlp.down_proj",
            5632,
            3.453325158646362e-22,
        ],
        ["Llama-2-7B", "model.layers.12.mlp.down_proj", 11008, 2.8546715776898906e-21],
        ["Llama-2-13B", "model.layers.12.mlp.down_proj", 13824, 5.291081534353017e-21],
        ["Llama-2-70B", "model.layers.12.mlp.down_proj", 28672, 4.397710435251772e-22],
        # 16
        [
            "TinyLlama-1.1B",
            "model.layers.16.mlp.down_proj",
            5632,
            1.3587203942714328e-21,
        ],
        ["Llama-2-7B", "model.layers.16.mlp.down_proj", 11008, 5.940289246362384e-21],
        ["Llama-2-13B", "model.layers.16.mlp.down_proj", 13824, 7.309526409850914e-21],
        ["Llama-2-70B", "model.layers.16.mlp.down_proj", 28672, 1.030444324421089e-20],
        # 20
        [
            "TinyLlama-1.1B",
            "model.layers.20.mlp.down_proj",
            5632,
            1.5188557563370398e-21,
        ],
        ["Llama-2-7B", "model.layers.20.mlp.down_proj", 11008, 1.1183485469540252e-20],
        ["Llama-2-13B", "model.layers.20.mlp.down_proj", 13824, 1.108525197693666e-20],
        ["Llama-2-70B", "model.layers.20.mlp.down_proj", 28672, 1.6725920123638545e-20],
    ]

    df = pd.DataFrame(
        raw_data,
        columns=["model", "layer", "rxx_size", "error_ratio"],
    )

    print(df)
    plot_palette("cz")

    # plotting arguments
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_yscale("log")
    # plt.plot(
    #     df.loc[df["layer"] == "model.layers.1.mlp.down_proj", "rxx_size"],
    #     df.loc[df["layer"] == "model.layers.1.mlp.down_proj", "error_ratio"],
    #     "o--",
    #     markersize=markersize,
    #     color=get_cz_color("cz_orange"),
    #     label="layer 1",
    # )
    plt.plot(
        df.loc[df["layer"] == "model.layers.4.mlp.down_proj", "rxx_size"],
        df.loc[df["layer"] == "model.layers.4.mlp.down_proj", "error_ratio"],
        "o--",
        markersize=markersize,
        color=get_color("cbf_green"),
        label="layer 4",
    )
    plt.plot(
        df.loc[df["layer"] == "model.layers.8.mlp.down_proj", "rxx_size"],
        df.loc[df["layer"] == "model.layers.8.mlp.down_proj", "error_ratio"],
        "o--",
        markersize=markersize,
        color=get_color("cbf_red"),
        label="layer 8",
    )
    plt.plot(
        df.loc[df["layer"] == "model.layers.12.mlp.down_proj", "rxx_size"],
        df.loc[df["layer"] == "model.layers.12.mlp.down_proj", "error_ratio"],
        "o--",
        markersize=markersize,
        label="layer 12",
        color=get_color("cbf_purple"),
    )
    plt.plot(
        df.loc[df["layer"] == "model.layers.16.mlp.down_proj", "rxx_size"],
        df.loc[df["layer"] == "model.layers.16.mlp.down_proj", "error_ratio"],
        "o--",
        markersize=markersize,
        color=get_color("cbf_blue"),
        label="layer 16",
    )
    plt.plot(
        df.loc[df["layer"] == "model.layers.20.mlp.down_proj", "rxx_size"],
        df.loc[df["layer"] == "model.layers.20.mlp.down_proj", "error_ratio"],
        "o--",
        markersize=markersize,
        color=get_color("cbf_orange"),
        label="layer 20",
    )
    ax.set_ylabel(
        r"$R_{xx}$ error ratio $\log (\frac{\| R_{xx}\ Error \|_F}{\| R_{xx} \|_F})$"
    )
    ax.set_xlabel(r"Layer hidden size")
    ax.set_xticks(df["rxx_size"].unique())
    ax.set_xticklabels(["1.1B", "7B", "13B", "70B"])
    ax.legend(loc="upper left")
    ax.set_ylim([1e-23, 0.2e-18])

    plt.savefig("sqrtm_error_vs_rxx_size.pdf", bbox_inches="tight")


visualize_sqrtm_error_vs_rxx_size()
