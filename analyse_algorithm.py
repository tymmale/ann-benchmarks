import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file_path = 'results_complete.csv'
    data = pd.read_csv(file_path)

    columns_of_interest = ["p50", "p95", "p99", "p999", "qps", "build", "indexsize", "k-nn", "epsilon", "largeepsilon"]
    parameters = ["algorithm"]

    data = data.dropna(subset=columns_of_interest)
    data = data[parameters + columns_of_interest]

    grouped_data = data.groupby(parameters)[columns_of_interest].median()

    # Reset index to facilitate plotting
    grouped_data = grouped_data.reset_index()

    algorithm_list = []
    for entry in grouped_data[parameters].to_dict("list").values():
        for datum in entry:
            algorithm_list.append(datum)
    result_list = []
    for entry in grouped_data[columns_of_interest].to_dict("list").values():
        result_list.append(entry)

    matplotlib.rc("xtick", labelsize=10)
    matplotlib.rc("ytick", labelsize=10)

    fig, ax = plt.subplots(figsize=(20, 10))

    for entry_index, entry in enumerate(result_list):
        for index, datum in enumerate(entry):
            plt.bar(algorithm_list[index], datum, color="tab:blue", zorder=10)

        plot_name = columns_of_interest[entry_index]

        plt.grid(True, axis="y", zorder=0)
        plt.ylabel(f"{plot_name}", fontsize="15")
        plt.xlabel("Implementations", fontsize="15")
        plt.suptitle(f"{plot_name}", y=1.02)

        plt.savefig(f"analyse_algorithm_plots/benchmark_v_4_0_{plot_name}.png")
        plt.clf()

    plt.close()

