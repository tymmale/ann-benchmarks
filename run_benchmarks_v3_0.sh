PYTHON=${PY:="python3 -u"}
PARALLEL=${PARALLELISM:=1}
GISTPARALLEL=${GISTPARALLELISM:=$PARALLEL}

ALGODEF=${DEF:=custom_definitions_first_run}

start=`date +%s`
echo Started benchmark at `date`

mkdir -p logs

for ds in glove-50-angular glove-100-angular glove-200-angular sift-128-euclidean nytimes-256-angular nytimes-16-angular fashion-mnist-784-euclidan; do
	for algo in vespa-hnsw pgvector-hnsw pgvector-ivfflat elasticsearch-hnsw weaviate-hnsw qdrant-hnsw redisearch-hnsw; do
		$PYTHON run.py --dataset $ds --definitions $ALGODEF --runs 3 --count 2 4 10 20 50 100 --parallelism $PARALLEL --algorithm $algo
		cp annb.log logs/${ds}_parallel_${PARALLEL}_second_run.log
	done

	# separate milvus runs because problems with parallelism
	for algo in milvus-flat milvus-ivfflat milvus-ivfsq8 milvus-ivfpq milvus-hnsw milvus-scann; do
		$PYTHON run.py --dataset $ds --definitions $ALGODEF --runs 3 --count 2 4 10 20 50 100 --parallelism 1 --algorithm $algo
		cp annb.log logs/${ds}_parallel_1_second_run.log
	done
done

for algo in milvus-flat milvus-ivfflat milvus-ivfsq8 milvus-ivfpq milvus-hnsw milvus-scann; do
	$PYTHON run.py --dataset gist-960-euclidean --definitions $ALGODEF --runs 1 --count 2 4 10 20 50 100 --parallelism 1 --algorithm $algo
	cp annb.log logs/gist-960-euclidean_parallel_1_second_run.log
done

for algo in vespa-hnsw pgvector-hnsw pgvector-ivfflat elasticsearch-hnsw weaviate-hnsw qdrant-hnsw redisearch-hnsw; do
	$PYTHON run.py --dataset gist-960-euclidean --definitions $ALGODEF --runs 1 --count 2 4 10 20 50 100 --parallelism $GISTPARALLEL --algorithm $algo
	cp annb.log logs/gist-960-euclidean_parallel_${GISTPARALLEL}_second_run.log
done

end=`date +%s`
echo Finished benchmark at `date`

echo Spent $((end- start))s benchmarking.




