PYTHON=${PY:="python3 -u"}
PARALLEL=${PARALLELISM:=1}
GISTPARALLEL=${GISTPARALLELISM:=$PARALLEL}

ALGODEF=${DEF:=custom_definitions_first_run}

start=`date +%s`
echo Started benchmark at `date`

mkdir -p logs

for k in 2 4 10 20 50 100; do
  for ds in glove-50-angular glove-100-angular glove-200-angular sift-128-euclidean nytimes-256-angular nytimes-16-angular fashion-mnist-784-euclidan; do
    for algo in vespa-hnsw pgvector-hnsw pgvector-ivfflat elasticsearch-hnsw weaviate-hnsw qdrant-hnsw redisearch-hnsw; do
        $PYTHON run.py --dataset $ds --definitions $ALGODEF --runs 3 --count $k --parallelism $PARALLEL --algorithm $algo
        cp annb.log logs/${ds}_${k}_parallel_${PARALLEL}.log
    done
    # separate milvus runs because problems with parallelism
    for algo in milvus-flat milvus-ivfflat milvus-ivfsq8 milvus-ivfpq milvus-hnsw milvus-scann; do
      $PYTHON run.py --dataset $ds --definitions $ALGODEF --runs 3 --count $k --parallelism 1 --algorithm $algo
      cp annb.log logs/${ds}_${k}_parallel_1.log
    done
  done
  for algo in milvus-flat milvus-ivfflat milvus-ivfsq8 milvus-ivfpq milvus-hnsw milvus-scann; do
    $PYTHON run.py --dataset gist-960-euclidean --definitions $ALGODEF --runs 1 --count $k --parallelism 1 --algorithm $algo
    cp annb.log logs/gist-960-euclidean_${k}_parallel_1.log
  done
  for algo in vespa-hnsw pgvector-hnsw pgvector-ivfflat elasticsearch-hnsw weaviate-hnsw qdrant-hnsw redisearch-hnsw; do
      $PYTHON run.py --dataset gist-960-euclidean --definitions $ALGODEF --runs 1 --count $k --parallelism $GISTPARALLEL --algorithm $algo
      cp annb.log logs/gist-960-euclidean_${k}_parallel_${GISTPARALLEL}.log
  done
done

end=`date +%s`
echo Finished benchmark at `date`

echo Spent $((end- start))s benchmarking.




