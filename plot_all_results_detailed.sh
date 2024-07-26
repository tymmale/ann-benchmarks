X=${XAXIS:="k-nn"}
Y=${YAXIS:="qps"}
XS=${XSCALE:="linear"}
YS=${YSCALE:="linear"}

for ds in glove-50-angular glove-100-angular glove-200-angular sift-128-euclidean; do
  mkdir -p results/${ds}_plots/detailed/${X}_${XS}_${Y}_${YS}
  for count in 2 4 10 20 50 100; do
      python3 plot_detailed_configurations.py --dataset $ds --count $count -x $X -y $Y -X $XS -Y $YS \
       -o results/${ds}_plots/detailed/${X}_${XS}_${Y}_${YS}/${ds}_x_${X}_y_${Y}_xs_${XS}_ys_${YS}_k_${count}.png
  done
done
