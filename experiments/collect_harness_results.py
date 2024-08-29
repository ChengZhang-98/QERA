from pathlib import Path
import yaml
import argparse
import re

import pandas as pd


TASK_NAMES = [
    "arc_challenge",
    "boolq",
    "commonsense_qa",
    "leaderboard_bbh",
    "leaderboard_bbh_boolean_expressions",
    "leaderboard_bbh_causal_judgement",
    "leaderboard_bbh_date_understanding",
    "leaderboard_bbh_disambiguation_qa",
    "leaderboard_bbh_formal_fallacies",
    "leaderboard_bbh_geometric_shapes",
    "leaderboard_bbh_hyperbaton",
    "leaderboard_bbh_logical_deduction_five_objects",
    "leaderboard_bbh_logical_deduction_seven_objects",
    "leaderboard_bbh_logical_deduction_three_objects",
    "leaderboard_bbh_movie_recommendation",
    "leaderboard_bbh_navigate",
    "leaderboard_bbh_object_counting",
    "leaderboard_bbh_penguins_in_a_table",
    "leaderboard_bbh_reasoning_about_colored_objects",
    "leaderboard_bbh_ruin_names",
    "leaderboard_bbh_salient_translation_error_detection",
    "leaderboard_bbh_snarks",
    "leaderboard_bbh_sports_understanding",
    "leaderboard_bbh_temporal_sequences",
    "leaderboard_bbh_tracking_shuffled_objects_five_objects",
    "leaderboard_bbh_tracking_shuffled_objects_seven_objects",
    "leaderboard_bbh_tracking_shuffled_objects_three_objects",
    "leaderboard_bbh_web_of_lies",
    "leaderboard_gpqa",
    "leaderboard_gpqa_diamond",
    "leaderboard_gpqa_extended",
    "leaderboard_gpqa_main",
    "leaderboard_mmlu_pro",
    "loqer_benchmark_classic",
    "loqer_benchmark_hard",
    "wikitext",
    "winogrande",
]

TASKS_TO_MERGE = {
    "leaderboard_bbh": "bigbench-hard",
    "leaderboard_gpqa": "gpqa",
}

TASKS_TO_IGNORE = [
    "loqer_benchmark_classic",
    "loqer_benchmark_hard",
    "leaderboard_bbh",
    "leaderboard_gpqa",
]

METRICS_TO_COLLECT = {
    "arc_challenge": "acc_norm,none",
    "boolq": "acc,none",
    "commonsense_qa": "acc,none",
    r"leaderboard_bbh_.+": "acc_norm,none",
    r"leaderboard_gpqa_.+": "acc_norm,none",
    "leaderboard_mmlu_pro": "acc,none",
    "wikitext": "word_perplexity,none",
    "winogrande": "acc,none",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lm_eval_results_yaml", type=str)
    parser.add_argument("--output-csv", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.lm_eval_results_yaml, "r") as f:
        results = yaml.safe_load(f)

    results = results["results"]

    for task_name_pattern in TASKS_TO_IGNORE:
        results.pop(task_name_pattern, None)

    df = pd.DataFrame(columns=["task", "metric", "value"])

    merged_results = {}
    for task_name_pattern, metric in METRICS_TO_COLLECT.items():
        for task in results:
            if re.match(task_name_pattern, task):
                if task_name_pattern not in merged_results:
                    merged_results[task_name_pattern] = {"value": 0, "count": 0}

                merged_results[task_name_pattern]["value"] += results[task][metric]
                merged_results[task_name_pattern]["count"] += 1

    for task_name_pattern, metric in merged_results.items():
        task_name_pattern = TASKS_TO_MERGE.get(task_name_pattern, task_name_pattern)
        df.loc[len(df)] = [task_name_pattern, METRICS_TO_COLLECT[task_name_pattern], metric["value"] / metric["count"]]

    print(df)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
