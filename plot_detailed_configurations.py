import matplotlib as mpl

mpl.use("Agg")  # noqa
import argparse

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils_detailed_configurations import (compute_metrics, create_linestyles,
                                                                   create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results


def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, batch):
    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    for algo in sorted(all_data.keys(), key=mean_y):
        xm, ym = (metrics[xn], metrics[yn])
        # Now generate each plot
        handles = []
        labels = []
        plt.figure(figsize=(12, 9))

        # Find range for logit x-scale
        min_x, max_x = 1, 0

        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        linestyles = create_linestyles(als)
        # Print all results separately
        for entry_x, entry_y, algo_name in zip(axs, ays, als):
            color, faded, linestyle, marker = linestyles[algo_name]
            (handle,) = plt.plot(
                entry_x, entry_y, "-", label=algo_name, ms=7, mew=3, lw=2, marker=marker
            )

            handles.append(handle)
            labels.append(algo_name)

        ax = plt.gca()
        ax.set_ylabel(ym["description"])
        ax.set_xlabel(xm["description"])
        # Custom scales of the type --x-scale a3
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
                # plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
                plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
        # Other x-scales
        else:
            ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_title(get_plot_label(xm, ym))
        plt.gca().get_position()
        # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9}, ncol=2)
        plt.grid(visible=True, which="major", color="0.65", linestyle="-")
        plt.setp(ax.get_xminorticklabels(), visible=True)

        # Logit scale has to be a subset of (0,1)
        if "lim" in xm and x_scale != "logit":
            x0, x1 = xm["lim"]
            plt.xlim(max(x0, 0), min(x1, 1))
        elif x_scale == "logit":
            plt.xlim(min_x, max_x)
        if "lim" in ym:
            plt.ylim(ym["lim"])

        # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
        ax.spines["bottom"]._adjust_location()
        output_path = fn_out.split(".")
        output_path = f"_{algo}.".join(output_path)

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular")
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output")
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
        args.output = "results/%s.png" % (args.dataset + ("-batch" if args.batch else ""))
        print("writing output to %s" % args.output)

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, args.batch)
    runs = compute_metrics(np.array(dataset["distances"]), results, args.x_axis, args.y_axis, args.recompute)

    if not runs:
        raise Exception("Nothing to plot")

    create_plot(
        runs, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, args.batch
    )
