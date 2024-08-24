import matplotlib as mpl

mpl.use("Agg")  # noqa
import argparse

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils_plot_by_algorithm import (compute_metrics, create_linestyles,
                                                             create_pointset, get_plot_label)
from ann_benchmarks.results_by_algorithm import get_unique_algorithms, load_all_results

def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, batch):
    xm, ym = (metrics[xn], metrics[yn])
    labels = []
    handles = []
    plt.figure(figsize=(12, 9))

    metrics_to_filter = ["k-nn", "epsilon", "largeepsilon"]

    def mean_y(algo):
        xs, ys, ls, axs, ays, als, dataset = create_pointset(datum[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    min_x, max_x = 1, 0
    # Sorting by mean y-value helps aligning plots with labels

    for datum in all_data:
        algorithms = sorted(datum.keys(), key=mean_y)
        for algo in algorithms:
            xs, ys, ls, axs, ays, als, dataset = create_pointset(datum[algo], xn, yn)

            unique_dataset = set(dataset).pop()
            # Filter out values below 0.1
            if xn in metrics_to_filter and x_scale == "linear":
                filtered_indices = [i for i, x in enumerate(xs) if x >= 0.1]
                xs = [xs[i] for i in filtered_indices]
                ys = [ys[i] for i in filtered_indices]
                ls = [ls[i] for i in filtered_indices]
            elif yn in metrics_to_filter and y_scale == "linear":
                filtered_indices = [i for i, y in enumerate(ys) if y >= 0.1]
                ys = [ys[i] for i in filtered_indices]
                xs = [xs[i] for i in filtered_indices]
                ls = [ls[i] for i in filtered_indices]
            if "indexsize" in yn:
                filtered_indices = [i for i, y in enumerate(ys) if y >= 0.0]
                ys = [ys[i] for i in filtered_indices]
                xs = [xs[i] for i in filtered_indices]
                ls = [ls[i] for i in filtered_indices]

            min_x = min([min_x] + [x for x in xs if x > 0])
            max_x = max([max_x] + [x for x in xs if x < 1])
            color, faded, linestyle, marker = linestyles[unique_dataset]
            (handle,) = plt.plot(
                xs, ys, "-", label=algo, color=color, ms=7, mew=3, lw=3, marker=marker
            )
            handles.append(handle)

            if raw:
                if "k-nn" == xn and x_scale == "linear":
                    filtered_indices = [i for i, x in enumerate(axs) if x >= 0.1]
                    axs = [axs[i] for i in filtered_indices]
                    ays = [ays[i] for i in filtered_indices]
                    als = [als[i] for i in filtered_indices]
                elif "k-nn" == yn and y_scale == "linear":
                    filtered_indices = [i for i, y in enumerate(ays) if y >= 0.1]
                    ays = [ays[i] for i in filtered_indices]
                    axs = [axs[i] for i in filtered_indices]
                    als = [als[i] for i in filtered_indices]
                if "indexsize" in yn:
                    filtered_indices = [i for i, y in enumerate(ays) if y >= 0.0]
                    ays = [ays[i] for i in filtered_indices]
                    axs = [axs[i] for i in filtered_indices]
                    als = [als[i] for i in filtered_indices]

                plt.plot(
                    axs, ays, "-", label=algo, color=faded, ms=5, mew=2, lw=2, marker=marker
                )
            labels.append(unique_dataset)

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])

    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def fun(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_fun(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(f"{algo_name} performance: {get_plot_label(xm, ym)}" if algo_name else get_plot_label(xm, ym))
    plt.gca().get_position()
    # ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9})
    # If legends should be beneath the plot
    ax.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, -0.15), prop={"size": 9}, ncol=(len(labels)))
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if "lim" in ym:
        plt.ylim(ym["lim"])

    if "k-nn" == xn and x_scale == "linear":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0.1), min(x1, 1))
    if "k-nn" == yn and y_scale == "linear":
        y0, y1 = ym["lim"]
        plt.ylim(max(y0, 0.1), min(y1, 1))

    ax.spines["bottom"]._adjust_location()

    plt.savefig(fn_out, bbox_inches="tight", pad_inches=0.2)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", metavar="DATASET", nargs="*", default=["glove-100-angular"])
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output", help="Output directory for plots", default="results/")
    parser.add_argument(
        "-x", "--x-axis", help="Which metric to use on the X-axis", choices=metrics.keys(), default="k-nn"
    )
    parser.add_argument(
        "-y", "--y-axis", help="Which metric to use on the Y-axis", choices=metrics.keys(), default="qps"
    )
    parser.add_argument(
        "-X", "--x-scale", help="Scale to use when drawing the X-axis. Typically linear, logit or a2", default="linear"
    )
    parser.add_argument(
        "-Y",
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )
    parser.add_argument("--batch", help="Plot runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    args = parser.parse_args()

    if not args.output:
        dataset_names = "_".join(args.datasets)
        args.output = f"results/{dataset_names}"
        print("writing output to %s" % args.output)

    datasets = []
    for entry in args.datasets:
        dataset, _ = get_dataset(entry)
        datasets.append(dataset)

    count = int(args.count)
    unique_algorithms = get_unique_algorithms()

    results = [[] for _ in unique_algorithms]
    for index, algo_name in enumerate(unique_algorithms):
        for ds_name in args.datasets:
            results[index].append(load_all_results(ds_name, count, algo_name, args.batch))

    runs = [[] for _ in unique_algorithms]
    for index in range(len(unique_algorithms)):
        for ds_index, ds in enumerate(datasets):
            runs[index].append(compute_metrics(np.array(ds["distances"]), results[index][ds_index], args.x_axis, args.y_axis, args.recompute))
        if not runs:
            raise Exception("Nothing to plot")

    for entry, algo_name in zip(runs, unique_algorithms):
        linestyles = create_linestyles(args.datasets)
        output_file = f"{args.output}_{algo_name}.png"
        create_plot(
            entry, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, output_file, linestyles, args.batch,
        )
