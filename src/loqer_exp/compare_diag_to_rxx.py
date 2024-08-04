import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loqer.ptq_pipeline import _merge_chunked_approximation_error
from loqer.logging import get_logger

logger = get_logger(__name__)


def _compare_diag_and_rxx(df_diag: Path, df_rxx: Path):
    """
    Compare the mse of diag and rxx for each layer. Plot a histogram of the comparison.
    """

    df_merged = pd.merge(df_diag, df_rxx, on="layer_name", suffixes=("_diag", "_rxx"))
    df_merged["rxx<=diag?"] = df_merged["mse_rxx"] <= df_merged["mse_diag"]
    df_merged["layer_tag"] = df_merged["layer_name"].apply(lambda x: x.split(".")[-1])
    df_merged["mse_diag-mse_rxx"] = df_merged["mse_diag"] - df_merged["mse_rxx"]

    df_concise = df_merged.loc[:, ["layer_tag", "rxx<=diag?"]]
    fig, ax = plt.subplots()
    palette = sns.color_palette("coolwarm", n_colors=2)
    palette.reverse()
    sns.histplot(data=df_concise, x="layer_tag", hue="rxx<=diag?", ax=ax, multiple="stack", palette=palette)
    plt.xticks(rotation=45)
    # draw a horizontal line in the middle
    num_types = len(df_concise["layer_tag"].unique())
    num_decoder_layers = len(df_concise) / num_types
    plt.axhline(num_decoder_layers / 2, color="red", linestyle="--")
    return df_merged, fig


def cli_compare_diag_and_rxx():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", "-d", dest="diag", type=str, required=True)
    parser.add_argument("--rxx", "-r", dest="rxx", type=str, required=True)
    parser.add_argument("--quick-save", "-q", dest="quick_save", action="store_true", default=False)
    parser.add_argument("--save-name", "-s", dest="save_name", type=str, default=None)
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true", default=False)

    args = parser.parse_args()

    diag_path = Path(args.diag)
    rxx_path = Path(args.rxx)
    assert diag_path.exists()
    assert rxx_path.exists()

    if diag_path.is_dir():
        logger.info(f"diag_path is a directory. Merging chunked approximation error files.")
        df_diag = _merge_chunked_approximation_error(diag_path)
    else:
        df_diag = pd.read_csv(diag_path)

    if rxx_path.is_dir():
        logger.info(f"rxx_path is a directory. Merging chunked approximation error files.")
        df_rxx = _merge_chunked_approximation_error(rxx_path)
    else:
        df_rxx = pd.read_csv(rxx_path)

    df, fig = _compare_diag_and_rxx(df_diag=df_diag, df_rxx=df_rxx)

    if args.quick_save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name = f"quick-save-{timestamp}"
        df.to_csv(f"{save_name}.csv", index=False)
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight")
        logger.info(f"Quick saved to {save_name}.csv and {save_name}.pdf")

    if args.save_name is not None:
        save_name_csv = Path(args.save_name + ".csv")
        save_name_pdf = Path(args.save_name + ".pdf")
        save_name_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_name_csv, index=False)
        fig.savefig(save_name_pdf, bbox_inches="tight")
        logger.info(f"Saved to {save_name_csv} and {save_name_pdf}")

    df_pretty = df.sort_values(by=["rxx<=diag?", "layer_tag", "mse_diag-mse_rxx"], ascending=[False, True, True])
    df_pretty["rxx<=diag?"] = df_pretty["rxx<=diag?"].map({True: "✅", False: "⚠️"})
    if args.verbose:
        logger.info(f"df for comparison:\n{df_pretty.to_markdown()}")

    df_concise = df.loc[:, ["layer_tag", "rxx<=diag?"]]
    logger.info(f"summary of df for comparison:\n{df_concise.groupby('layer_tag').agg('mean')}")
