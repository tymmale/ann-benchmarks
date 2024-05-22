import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN


def start_postgres():
    try:
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

        print("[PostgreSQL] PotsgreSQL service has been started!")
    except Exception as e:
        print(f"[PostgreSQL]  PotsgreSQL service could not be started: {e}!")


def stop_postgres():
    try:
        subprocess.run("service postgresql stop", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

        print("[PostgreSQL] PotsgreSQL service has been stopped!")
    except Exception as e:
        print(f"[PostgreSQL] PostgreSQL service could not be stopped: {e}!")


class PGVector(BaseANN):
    _connection_string = """
    dbname=ann
    user=ann
    password=ann
"""

    def __init__(self, metric, index_param):
        self._metric = metric

        start_postgres()

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)
            start_postgres()

            conn.execute("DROP TABLE IF EXISTS items")
            conn.execute(f"CREATE TABLE items (id int, embedding vector({X.shape[1]}))")
            conn.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
            print("copying data...")

            cur = conn.cursor()
            with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:

                copy.set_types(["int4", "vector"])
                for i, embedding in enumerate(X):
                    copy.write_row((i, embedding))

            print("creating index...")
            conn.execute(self.get_index_param())
            print("done!")

    def set_query_arguments(self, ef_search):
        raise NotImplementedError()

    def query(self, v, n):
        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)

            result = conn.execute(self._query, (v, n), binary=True, prepare=True)
            return [index for index, in result.fetchall()]

    def get_memory_usage(self):
        try:
            with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
                pgvector.psycopg.register_vector(conn)

                result = conn.execute("SELECT pg_relation_size('items_embedding_idx');")
                return result.fetchone()[0] / 1024
        except Exception as e:
            print(f"[PostgreSQL] Memory usage could not be fetched: {e}")
            return 0

    def done(self):
        # Stop PostgreSQL service
        stop_postgres()

    def get_index_param(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class PGVectorIVFFLAT(PGVector):
    def __init__(self, metric, index_param):
        super().__init__(self, metric, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        if self._metric == "angular":
            return (f"CREATE INDEX items_embedding_idx ON items USING ivfflat (embedding vector_cosine_ops) "
                    f"WITH (lists = {self._index_nlist}")

        elif self._metric == "euclidean":
            return (f"CREATE INDEX items_embedding_idx ON items USING ivfflat (embedding vector_l2_ops) "
                    f"WITH (lists = {self._index_nlist}")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def set_query_arguments(self, nrpobe):
        self.nprobe = nrpobe

        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)

            conn.execute(f"SET ivfflat.probes {nrpobe}")

    def __str__(self):
        return f"PGVector metric:{self._metric} index_nlist:{self._index_nlist}"


class PGVectorHSNW(PGVector):
    def __init__(self, metric, index_param):
        super().__init__(self, metric, index_param)

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
        return (f"PGVector metric:{self._metric}, index_m:{self._index_m}, index_ef:{self._index_ef}, "
                f"search_ef:{self._ef_search}")
