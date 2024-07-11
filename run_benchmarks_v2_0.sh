PYTHON=${PY:="python3 -u"}
PARALLEL=${PARALLELISM:=1}
GISTPARALLEL=${GISTPARALLELISM:=$PARALLEL}

# K=10 runs
ALGODEF=custom_definitions

start=`date +%s`
echo Started benchmark at `date`

mkdir -p logs

for ds in glove-50-angular glove-100-angular glove-200-angular sift-128-euclidean nytimes-256-angular nytimes-16-angular fashion-mnist-784-euclidan; do
  for algo in pgvector-hnsw vespa-hnsw pgvector-hnsw pgvector-ivfflat elasticsearch-hnsw weaviate-hnsw qdrant-hnsw redisearch-hnsw; do
      $PYTHON run.py --dataset $ds --definitions $ALGODEF --runs 3 --count 10 --parallelism $PARALLEL --algorithm $algo
      cp annb.log logs/${ds}_10_all.log
  done
done

# separate milvus runs because problems with parallelism
for ds in glove-50-angular glove-100-angular glove-200-angular sift-128-euclidean nytimes-256-angular nytimes-16-angular fashion-mnist-784-euclidan; do
  for algo in milvus-flat milvus-ivfflat milvus-ivfsq8 milvus-ivfpq milvus-hnsw milvus-scann; do
      $PYTHON run.py --dataset $ds --definitions $ALGODEF --runs 3 --count 10 --parallelism 1 --algorithm $algo
      cp annb.log logs/${ds}_10_milvus.log
  done
done

# separate GIST runs because of high RAM usage
for algo in pgvector-hnsw vespa-hnsw pgvector-hnsw pgvector-ivfflat elasticsearch-hnsw wewaviate-hnsw qdrant-hnsw redisearch-hnsw; do
  $PYTHON run.py --dataset gist-960-euclidean --definitions $ALGODEF --runs 1 --count 10 --parallelism $GISTPARALLEL --algorithm $algo
  cp annb.log logs/gist-960-euclidean_10_all.log
done

# separate milvus runs because problems with parallelism
for algo in milvus-flat milvus-ivfflat milvus-ivfsq8 milvus-ivfpq milvus-hnsw milvus-scann; do
  $PYTHON run.py --dataset gist-960-euclidean --definitions $ALGODEF --runs 1 --count 10 --parallelism 1 --algorithm $algo
  cp annb.log logs/gist-960-euclidean_10_milvus.log
done

end=`date +%s`
echo Finished benchmark at `date`

echo Spent $((end- start))s benchmarking.




