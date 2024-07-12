for ds in glove-50-angular glove-100-angular glove-200-angular sift-128-euclidean gist-960-euclidean nytimes-256-angular nytimes-16-angular fashion-mnist-784-euclidean; do
    python3 plot.py --dataset $ds --count 10
done
