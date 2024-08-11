X=${XAXIS:="k-nn"}
Y=${YAXIS:="qps"}
XS=${XSCALE:="linear"}
YS=${YSCALE:="linear"}

DATASETS=()

mkdir -p results/by_algorithm/${X}_${XS}_${Y}_${YS}
for count in 2 4 10 20 50 100; do
    python3 plot_by_algorithm.py --dataset glove-50-angular glove-100-angular nytimes-256-angular sift-128-euclidean fashion-mnist-784-euclidean \
     --count $count -x $X -y $Y -X $XS -Y $YS -o results/by_algorithm/${X}_${XS}_${Y}_${YS}/x_${X}_y_${Y}_xs_${XS}_ys_${YS}_k_${count}
done
