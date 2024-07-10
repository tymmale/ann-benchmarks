import subprocess
import sys
import time

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN


class PGVector(BaseANN):
    _connection_string = """
    dbname=ann
    user=ann
    password=ann
"""

    def __init__(self, metric, index_param):
        self._metric = metric

        self._conn = None
        self._cur = None

        self._is_running = False

        self.start_postgres()

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def start_postgres(self):
        try:
            if not self._is_running:
                subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
                self._is_running =True

                print("[PostgreSQL] PotsgreSQL service has been started!")
            else:
                print("[PostgreSQL] PotsgreSQL service has already stopped. Doing nothing!")
        except Exception as e:
            print(f"[PostgreSQL]  PotsgreSQL service could not be started: {e}!")

    def stop_postgres(self):
        try:
            if self._is_running:
                subprocess.run("service postgresql stop", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
                self._is_running = False

                print("[PostgreSQL] PotsgreSQL service has been stopped!")
            else:
                print("[PostgreSQL] PotsgreSQL service has already stopped. Doing nothing!")
        except Exception as e:
            print(f"[PostgreSQL] PostgreSQL service could not be stopped: {e}!")

    def fit(self, X):
        self.start_postgres()

        conn = psycopg.connect(conninfo=self._connection_string, autocommit=True)
        pgvector.psycopg.register_vector(conn)

        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(f"CREATE TABLE items (id int, embedding vector({X.shape[1]}))")
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")

        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            start_time = time.time()

            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))

            end_time = time.time()

        print(f"[PostgreSQL] Copying data took {end_time - start_time} seconds.")

        print("creating index...")
        cur.execute(self.get_index_param())
        print("done!")

        self._conn = conn
        self._cur = cur

    def set_query_arguments(self, ef_search):
        raise NotImplementedError()

    def query(self, v, n):
        query_result = self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        results = [index for index, in query_result.fetchall()]

        return results

    def get_memory_usage(self):
        if self._cur is not None:
            result = self._cur.execute("SELECT pg_relation_size('items_embedding_idx');")
            return result.fetchone()[0] / 1024
        else:
            print(f"[PostgreSQL] Memory usage could not be fetched because cursor is None!")
            return 0

    def done(self):
        # Stop PostgreSQL service
        self.stop_postgres()

        # Close connections
        self._cur.close()
        self._conn.close()

    def get_index_param(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class PGVectorIVFFLAT(PGVector):
    def __init__(self, metric, index_param):
        super().__init__(metric, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        if self._metric == "angular":
            return (f"CREATE INDEX items_embedding_idx ON items USING ivfflat (embedding vector_cosine_ops) "
                    f"WITH (lists = {self._index_nlist})")

        elif self._metric == "euclidean":
            return (f"CREATE INDEX items_embedding_idx ON items USING ivfflat (embedding vector_l2_ops) "
                    f"WITH (lists = {self._index_nlist})")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def set_query_arguments(self, nprobe):
        self.nprobe = nprobe

        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)

            conn.execute(f"SET ivfflat.probes = {nprobe}")

    def __str__(self):
        return f"PGVectorIVFFLAT metric:{self._metric} nlist:{self._index_nlist}, nprobe:{self.nprobe}"


class PGVectorHNSW(PGVector):
    def __init__(self, metric, index_param):
        super().__init__(metric, index_param)

        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        if self._metric == "angular":
            return (f"CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {self._index_m}, ef_construction = {self._index_ef})")

        elif self._metric == "euclidean":
            return (f"CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_l2_ops) "
                    f"WITH (m = {self._index_m}, ef_construction = {self._index_ef})")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def set_query_arguments(self, ef):
        self._ef_search = ef

        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)

            conn.execute(f"SET hnsw.ef_search = {ef}")

    def __str__(self):
        return (f"PGVectorHSNW metric:{self._metric}, M:{self._index_m}, ef_construction:{self._index_ef}, "
                f"ef:{self._ef_search}")
