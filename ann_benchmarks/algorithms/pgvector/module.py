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

    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
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

            if self._metric == "angular":
                conn.execute(
                    f"CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_cosine_ops) WITH "
                    f"(m = {self._m}, ef_construction = {self._ef_construction})")
            elif self._metric == "euclidean":
                conn.execute(
                    f"CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_l2_ops) WITH (m = {self._m}, "
                    f"ef_construction = {self._ef_construction})")
            else:
                raise RuntimeError(f"unknown metric {self._metric}")
            print("done!")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search

        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)

            conn.execute(f"SET hnsw.ef_search = {ef_search}")

    def query(self, v, n):
        with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)

            result = conn.execute(self._query, (v, n), binary=True, prepare=True)
            return [index for index, in result.fetchall()]

    def done(self):
        # Stop PostgreSQL service
        stop_postgres()

    def get_memory_usage(self):
        try:
            with psycopg.connect(conninfo=self._connection_string, autocommit=True) as conn:
                pgvector.psycopg.register_vector(conn)

                result = conn.execute("SELECT pg_relation_size('items_embedding_idx');")
                return result.fetchone()[0] / 1024
        except Exception as e:
            print(f"[PostgreSQL] Memory usage could be fetched: {e}")
            return 0

    def __str__(self):
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
