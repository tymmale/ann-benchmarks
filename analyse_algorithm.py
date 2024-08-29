import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file_path = 'results_complete.csv'
    data = pd.read_csv(file_path)

    columns_of_interest = ["p50", "p95", "p99", "p999", "qps", "build", "indexsize", "k-nn", "epsilon", "largeepsilon"]

    y_labels_lookup_table = {"p50": "Milliseconds",
                             "p95": "Milliseconds",
                             "p99": "Milliseconds",
                             "p999": "Milliseconds",
                             "qps": "Queries per Second",
                             "build": "Seconds",
                             "indexsize": "MB",
                             "k-nn": "Recall",
                             "epsilon": "Epsilon 0,01 Recall",
                             "largeepsilon": "Epsilon 0,1 Recall"
                             }

    parameters = ["algorithm"]

    # Remove data with recall >= 0,1
    #data = data[data["k-nn"] >= 0.1]
    data = data.dropna(subset=columns_of_interest)
    data = data[parameters + columns_of_interest]

    grouped_data = data.groupby(parameters)[columns_of_interest].median()

    # Reset index to facilitate plotting
    grouped_data = grouped_data.reset_index()

    algorithm_list = []
    for entry in grouped_data[parameters].to_dict("list").values():
        for datum in entry:
            split_datum = datum.split("-")
            split_datum = "-\n".join(split_datum)
            algorithm_list.append(split_datum)

    result_list = []
    for entry in grouped_data[columns_of_interest].to_dict("list").values():
        result_list.append(entry)

    font = {"size": 15}
    matplotlib.rc("font", **font)

    fig, ax = plt.subplots(figsize=(20, 10))

    for entry_index, entry in enumerate(result_list):
        for index, datum in enumerate(entry):
            plt.bar(algorithm_list[index], datum, color="tab:blue", zorder=10)

        metric_name = columns_of_interest[entry_index]
        y_label = y_labels_lookup_table[metric_name]

        if "indexsize" in metric_name:
            plt.yscale("log")

        plt.grid(True, axis="y", zorder=0)
        plt.ylabel(f"{y_label}", wrap=True)
        plt.xlabel("\nImplementations")

        if "largeepsilon" in metric_name:
            plt.suptitle("Epsilon 0,1 Recall", y=1.02)
        elif "k-nn" in metric_name:
            plt.suptitle("Recall", y=1.02)
        else:
            plt.suptitle(f"{metric_name}", y=1.02)

        plt.savefig(f"analyse_algorithm_plots/benchmark_v_4_0_{metric_name}_median.png",
                    bbox_inches="tight", pad_inches=0.2)
        plt.clf()
