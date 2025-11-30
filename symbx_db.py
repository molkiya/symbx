# symbx_db.py
import time
import json
from typing import List, Optional, Dict, Any, Tuple

import psycopg2
import psycopg2.extras
import boto3

from symbx_rules import RULES, NAME2RULE, execute_program, canonical_form, program_hash

class SymbXDB:
    def __init__(self,
                 pg_dsn: str = "dbname=symbx user=symbx password=symbx host=127.0.0.1 port=5433",
                 s3_endpoint: str = "http://127.0.0.1:9000",
                 s3_access_key: str = "symbx",
                 s3_secret_key: str = "replicax12345",
                 s3_bucket: str = "symbx"):
        self.conn = psycopg2.connect(pg_dsn)
        self.conn.autocommit = True
        self.s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
        )
        self.bucket = s3_bucket

    # ---------- rules ----------
    def bootstrap_rules(self) -> None:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # insert rules if missing; create one active version per rule
            for r in RULES:
                cur.execute("SELECT rule_id FROM rule WHERE name=%s", (r.name,))
                row = cur.fetchone()
                if row:
                    rule_id = row["rule_id"]
                else:
                    cur.execute(
                        "INSERT INTO rule(name,family,parametric) VALUES(%s,%s,%s) RETURNING rule_id",
                        (r.name, r.family, r.parametric),
                    )
                    rule_id = cur.fetchone()["rule_id"]

                impl_digest = f"{r.family}:{r.name}"
                cur.execute(
                    """INSERT INTO rule_version(rule_id,impl_digest,params_schema,metadata,valid_from)
                       VALUES(%s,%s,%s,%s,now()) RETURNING rule_ver_id""",
                    (rule_id, impl_digest, json.dumps({}), json.dumps({}))
                )

    def resolve_rule_version_ids(self, prog_names: List[str]) -> List[int]:
        res = []
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            for name in prog_names:
                cur.execute("SELECT rule_id FROM rule WHERE name=%s", (name,))
                r = cur.fetchone()
                if not r:
                    raise RuntimeError(f"rule not found: {name}")
                cur.execute(
                    """SELECT rule_ver_id FROM rule_version
                       WHERE rule_id=%s AND valid_to IS NULL
                       ORDER BY valid_from DESC LIMIT 1""",
                    (r["rule_id"],)
                )
                v = cur.fetchone()
                if not v:
                    raise RuntimeError(f"rule version not found: {name}")
                res.append(v["rule_ver_id"])
        return res

    # ---------- programs ----------
    def upsert_program(self, prog_names: List[str], complexity: int = 0) -> Tuple[int, str]:
        canon = canonical_form(prog_names)
        phash = program_hash(canon)
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT program_id FROM program WHERE prog_hash=%s", (phash,))
            row = cur.fetchone()
            if row:
                program_id = row["program_id"]
            else:
                cur.execute(
                    """INSERT INTO program(prog_hash,canonical_form,length,complexity)
                       VALUES(%s,%s,%s,%s) RETURNING program_id""",
                    (phash, canon, len(prog_names), complexity)
                )
                program_id = cur.fetchone()["program_id"]
                # insert steps
                ver_ids = self.resolve_rule_version_ids(prog_names)
                for idx, ver_id in enumerate(ver_ids):
                    cur.execute(
                        """INSERT INTO program_step(program_id,step_idx,rule_ver_id,params)
                           VALUES(%s,%s,%s,%s)""",
                        (program_id, idx, ver_id, json.dumps({}))
                    )
        return program_id, phash

    # ---------- exec cache ----------
    def exec_with_cache(self, prog_names: List[str], a_value: float,
                        computed_by: str = "cpu") -> Tuple[float, float, str]:
        canon = canonical_form(prog_names)
        phash = program_hash(canon)

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT x_pred, mse FROM exec_cache WHERE prog_hash=%s AND a_value=%s",
                (phash, a_value)
            )
            row = cur.fetchone()
            if row:
                return float(row["x_pred"]), float(row["mse"]), phash

        t0 = time.time()
        x_pred = execute_program(a_value, prog_names)
        mse = 0.0  # placeholder if no target; store 0
        runtime_ms = int((time.time() - t0) * 1000)

        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO exec_cache(prog_hash,a_value,x_pred,mse,computed_by,runtime_ms)
                   VALUES(%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (prog_hash,a_value) DO NOTHING""",
                (phash, a_value, x_pred, mse, computed_by, runtime_ms)
            )
        return x_pred, mse, phash

    # ---------- episodes ----------
    def log_episode(self, experiment_id: int, task_id: Optional[int],
                    program_id: int, x_pred: float, mse: float,
                    reward: float, steps_count: int,
                    trace_uri: Optional[str] = None) -> int:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """INSERT INTO episode(experiment_id,task_id,program_id,x_pred,mse,reward,steps_count,trace_uri)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING episode_id""",
                (experiment_id, task_id, program_id, x_pred, mse, reward, steps_count, trace_uri)
            )
            return cur.fetchone()["episode_id"]
