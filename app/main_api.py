from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import psycopg2.extras

from symbx_db import SymbXDB
from symbx_rules import canonical_form

class Settings(BaseSettings):
    PG_DSN: str = "dbname=symbx user=symbx password=symbx host=127.0.0.1 port=5433"
    S3_ENDPOINT: str = "http://127.0.0.1:9000"
    S3_ACCESS_KEY: str = "symbx"
    S3_SECRET_KEY: str = "replicax12345"
    S3_BUCKET: str = "symbx"

settings = Settings()

db = SymbXDB(
    pg_dsn=settings.PG_DSN,
    s3_endpoint=settings.S3_ENDPOINT,
    s3_access_key=settings.S3_ACCESS_KEY,
    s3_secret_key=settings.S3_SECRET_KEY,
    s3_bucket=settings.S3_BUCKET,
)

app = FastAPI(title="SymbX API", version="0.1.0")

# --------- Schemas ---------
class ProgramUpsertReq(BaseModel):
    prog_names: List[str]
    complexity: int = 0

class ProgramUpsertResp(BaseModel):
    program_id: int
    prog_hash: str
    canonical_form: str
    length: int
    complexity: int

class ExecReq(BaseModel):
    prog_names: List[str]
    a_value: float
    computed_by: str = "cpu"

class ExecResp(BaseModel):
    prog_hash: str
    a_value: float
    x_pred: float
    mse: float

class EpisodeReq(BaseModel):
    experiment_id: int
    task_id: Optional[int] = None
    prog_names: List[str]
    x_pred: float
    mse: float
    reward: float
    steps_count: int
    trace_uri: Optional[str] = None
    complexity: int = 0

class EpisodeResp(BaseModel):
    episode_id: int

# --------- Endpoints ---------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rules/bootstrap")
def bootstrap_rules():
    db.bootstrap_rules()
    return {"status": "ok"}

@app.post("/programs", response_model=ProgramUpsertResp)
def upsert_program(req: ProgramUpsertReq):
    program_id, phash = db.upsert_program(req.prog_names, complexity=req.complexity)
    canon = canonical_form(req.prog_names)
    return ProgramUpsertResp(
        program_id=program_id,
        prog_hash=phash,
        canonical_form=canon,
        length=len(req.prog_names),
        complexity=req.complexity,
    )

@app.post("/execute", response_model=ExecResp)
def execute(req: ExecReq):
    x_pred, mse, phash = db.exec_with_cache(req.prog_names, req.a_value, computed_by=req.computed_by)
    return ExecResp(prog_hash=phash, a_value=req.a_value, x_pred=x_pred, mse=mse)

@app.get("/program/{phash}", response_model=ProgramUpsertResp)
def get_program(phash: str):
    # lightweight fetch via canonical form from DB
    with db.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT program_id, canonical_form, length, complexity FROM program WHERE prog_hash=%s", (phash,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "program not found")
        return ProgramUpsertResp(
            program_id=row["program_id"],
            prog_hash=phash,
            canonical_form=row["canonical_form"],
            length=row["length"],
            complexity=row["complexity"]
        )

@app.post("/episodes", response_model=EpisodeResp)
def log_episode(req: EpisodeReq):
    # ensure program exists (upsert)
    program_id, phash = db.upsert_program(req.prog_names, complexity=req.complexity)
    ep_id = db.log_episode(
        experiment_id=req.experiment_id,
        task_id=req.task_id,
        program_id=program_id,
        x_pred=req.x_pred,
        mse=req.mse,
        reward=req.reward,
        steps_count=req.steps_count,
        trace_uri=req.trace_uri,
    )
    return EpisodeResp(episode_id=ep_id)
