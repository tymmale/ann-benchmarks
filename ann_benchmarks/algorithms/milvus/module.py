import time
from time import sleep
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema, DataType, MilvusClient
import os

from ..base.module import BaseANN

MILVUS_URI = "http://127.0.0.1:19530"
MILVUS_DEFAULT_USER = "root"
MILVUS_DEFAULT_PASSWORD = "Milvus"
MILVUS_DEFAULT_DB = "default"


def metric_mapping(_metric: str):
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.start_milvus()
        self.connects = connections
        max_tries = 10
        for try_num in range(max_tries):
            try:
                self._client = MilvusClient(uri=MILVUS_URI,
                                            token=f"{MILVUS_DEFAULT_USER}:{MILVUS_DEFAULT_PASSWORD}",
                                            db_name=MILVUS_DEFAULT_DB
                                            )

                self.connects.connect(
                    alias="utility_connection",
                    host='localhost',
                    port='19530',
                    token=f"{MILVUS_DEFAULT_USER}:{MILVUS_DEFAULT_PASSWORD}"
                )
                break
            except Exception as e:
                if try_num == max_tries - 1:
                    raise Exception(f"[Milvus] connect to milvus failed: {e}!!!")
                print(f"[Milvus] try to connect to milvus again...")
                sleep(1)

        server_version = utility.get_server_version(using="utility_connection")
        print(f"[Milvus] Milvus version: {server_version}")

        self.collection_name = "test_milvus"
        if self._client.has_collection(self.collection_name):
            print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
            self._client.drop_collection(self.collection_name)

    def start_milvus(self) -> None:
        try:
            os.system("docker compose down")
            os.system("docker compose up -d")
            print("[Milvus] docker compose up successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose up failed: {e}!!!")

    def stop_milvus(self) -> None:
        try:
            os.system("docker compose down")
            print("[Milvus] docker compose down successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose down failed: {e}!!!")

    def create_collection(self) -> None:

        milvus_schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False, primary_field="id")
        milvus_schema.add_field(field_name="id", datatype=DataType.INT64)

        milvus_schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=milvus_schema,
            description="Test milvus search",
            kwargs={"consistence_level": "STRONG"}
        )

        # Use it to flush the collection
        self.collection = Collection(
            self.collection_name,
            milvus_schema,
            consistence_level="STRONG",
            using="utility_connection"
        )

        collection_description = self._client.describe_collection(self.collection_name)

        print(f"[Milvus] Create collection {collection_description} successfully!!!")

    def insert(self, X) -> None:
        # insert data
        print(f"[Milvus] Insert {len(X)} data into collection {self.collection_name}...")
        batch_size = 1000
        insertion_count = 0
        start_time = time.time()

        for i in range(0, len(X), batch_size):
            batch_data = X[i: min(i + batch_size, len(X))]
            entities = list(range(len(batch_data)))

            for index, entry in enumerate(batch_data):
                entities[index] = {
                    "id": min(index + i, len(X)),
                    "vector": entry
                }
            insertion_result = self._client.insert(self.collection_name, entities)
            insertion_count += insertion_result["insert_count"]

        end_time = time.time()
        # Seal all segments in the collection after data is inserted
        self.collection.flush(using="utility_connection")

        print(f"[Milvus] {insertion_count} data has been inserted into collection {self.collection_name}!!!")
        print(f"[Milvus] Inserting data took {end_time - start_time} seconds.")

    def get_index_param(self) -> dict:
        raise NotImplementedError()

    def create_index(self) -> None:
        # create index
        print(f"[Milvus] Create index for collection {self.collection_name}...")
        index_params = self._client.prepare_index_params()
        index_config = self.get_index_param()

        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type=index_config["index_type"],
            metric_type=index_config["metric_type"],
            params=index_config["params"]
        )
        self._client.create_index(self.collection_name, index_params)

        utility.wait_for_index_building_complete(
            collection_name=self.collection_name,
            index_name="vector_index",
            using="utility_connection"
        )

        index_progress = utility.index_building_progress(
            collection_name=self.collection_name,
            index_name="vector_index",
            using="utility_connection"
        )
        index_description = self._client.describe_index(self.collection_name, "vector_index")

        print(
            f"[Milvus] Create index {index_description} {index_progress} for collection {self.collection_name} "
            f"successfully!!!")

    def load_collection(self) -> None:
        # load collection
        print(f"[Milvus] Load collection {self.collection_name}...")
        self._client.load_collection(self.collection_name)

        utility.wait_for_loading_complete(self.collection_name, using="utility_connection")
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")

    def fit(self, X):
        self.create_collection()
        self.insert(X)
        self.create_index()
        self.load_collection()

    def query(self, v, n):
        results = self._client.search(
            collection_name=self.collection_name,
            data=[v],
            anns_field="vector",
            search_param=self.search_params,
            limit=n,
            output_fields=["id"]
        )

        ids = [r["id"] for r in results[0]]
        return ids

    def done(self):
        if self._client.has_collection(self.collection_name):
            self._client.release_collection(self.collection_name)
            self._client.drop_collection(self.collection_name)

        self.stop_milvus()


class MilvusFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type,
            "params": {}
        }

    def set_query_arguments(self, _ignore):
        self.search_params = {
            "index_type": "FLAT",
            "metric_type": self._metric_type,
        }

        self.name = f"MilvusFLAT metric:{self._metric}"


class MilvusIVFFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "index_type": "IVF_FLAT",
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFSQ8(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFPQ(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFPQ metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusHNSW(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, ef):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"MilvusHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class MilvusSCANN(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusDISKANN(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)

    def get_index_param(self):
        return {
            "index_type": "DISKANN",
            "metric_type": self._metric_type,
            "params": {}
        }

    def set_query_arguments(self, search_list):

        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"search_list": search_list}
        }

        self.name = f"MilvusDISKANN metric:{self._metric}, search_search_list={search_list}"
