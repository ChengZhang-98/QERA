from argparse import ArgumentParser
from pathlib import Path
import csv
import pandas as pd
from tabulate import tabulate

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("lm_eval_results_csv", type=str, nargs="+")
    parser.add_argument("--output-csv", "-o", dest="output_csv", type=str, default=None)
    args = parser.parse_args()

    paths = [Path(p) for p in args.lm_eval_results_csv]
    assert all(p.exists() for p in paths)

    headers = None
    rows = []

    for path in paths:
        df = pd.read_csv(path)
        if headers is None:
            headers = [["path"] + df.iloc[1, :].tolist(), ["-"] + df.iloc[2, :].tolist()]
        rows.append([path.as_posix()] + df.iloc[3, :].tolist())

    merged = headers + rows

    print(tabulate(merged, tablefmt="github"))

    if args.output_csv:
        # write to csv
        with open(args.output_csv, "w") as f:
            writer = csv.writer(f)
            writer.writerows(merged)