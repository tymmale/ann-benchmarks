import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


if __name__ == "__main__":

    file_path = "results_complete.csv"
    data = pd.read_csv(file_path)

    columns_of_interest = ["qps", "k-nn", "indexsize", "build"]
    parameters = ["nlist", "nprobe"]

    data["nlist"] = data["parameters"].str.extract(r"nlist:(\d+)").astype(float)
    data["nprobe"] = data["parameters"].str.extract(r"nprobe:(\d+)").astype(float)

    # Drop data point where nlist or nprobe are Nan and remove non
    data = data.dropna(subset=parameters)
    data = data[parameters + columns_of_interest]

    grouped_data = data.groupby(parameters).mean()[columns_of_interest]

    # Reset index to facilitate plotting
    grouped_data = grouped_data.reset_index()

    sns.pairplot(grouped_data, x_vars=parameters, y_vars=columns_of_interest, kind="scatter")
    plt.suptitle("Effect of 'nlist' and 'nprobe' on QPS, Recall (k-nn), Index Size, and Build Time", y=1.02)
    plt.savefig("nlist_nprobe_analysis_mean.png")
    plt.close()

    # Return the grouped data for further analysis
    grouped_data.head()
