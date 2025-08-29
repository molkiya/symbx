-- rules
CREATE TABLE IF NOT EXISTS rule (
  rule_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  family TEXT NOT NULL,
  parametric BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rule_version (
  rule_ver_id SERIAL PRIMARY KEY,
  rule_id INT NOT NULL REFERENCES rule(rule_id),
  impl_digest TEXT NOT NULL,
  params_schema JSONB,
  metadata JSONB,
  valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
  valid_to TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_rule_version_rule_id ON rule_version(rule_id);

-- programs
CREATE TABLE IF NOT EXISTS program (
  program_id BIGSERIAL PRIMARY KEY,
  prog_hash TEXT UNIQUE NOT NULL,
  canonical_form TEXT NOT NULL,
  length INT NOT NULL,
  complexity INT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS program_step (
  program_id BIGINT NOT NULL REFERENCES program(program_id) ON DELETE CASCADE,
  step_idx INT NOT NULL,
  rule_ver_id INT NOT NULL REFERENCES rule_version(rule_ver_id),
  params JSONB,
  PRIMARY KEY (program_id, step_idx)
);

-- datasets / tasks
CREATE TABLE IF NOT EXISTS dataset (
  dataset_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  generator TEXT,
  seed BIGINT,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task (
  task_id BIGSERIAL PRIMARY KEY,
  dataset_id INT REFERENCES dataset(dataset_id),
  a_value DOUBLE PRECISION NOT NULL,
  x_target DOUBLE PRECISION NOT NULL,
  true_prog_hash TEXT,
  metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_task_dataset ON task(dataset_id);
CREATE INDEX IF NOT EXISTS idx_task_trueph ON task(true_prog_hash);

-- models / experiments
CREATE TABLE IF NOT EXISTS model (
  model_id SERIAL PRIMARY KEY,
  arch TEXT NOT NULL,
  code_digest TEXT NOT NULL,
  weights_uri TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment (
  experiment_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  model_id INT REFERENCES model(model_id),
  config JSONB NOT NULL,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- episodes
CREATE TABLE IF NOT EXISTS episode (
  episode_id BIGSERIAL PRIMARY KEY,
  experiment_id INT REFERENCES experiment(experiment_id),
  task_id BIGINT REFERENCES task(task_id),
  policy_ver INT,
  program_id BIGINT REFERENCES program(program_id),
  x_pred DOUBLE PRECISION,
  mse DOUBLE PRECISION,
  reward DOUBLE PRECISION,
  steps_count INT,
  trace_uri TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_episode_experiment ON episode(experiment_id);
CREATE INDEX IF NOT EXISTS idx_episode_task ON episode(task_id);
CREATE INDEX IF NOT EXISTS idx_episode_program ON episode(program_id);

-- execution cache
CREATE TABLE IF NOT EXISTS exec_cache (
  cache_id BIGSERIAL PRIMARY KEY,
  prog_hash TEXT NOT NULL,
  a_value DOUBLE PRECISION NOT NULL,
  x_pred DOUBLE PRECISION NOT NULL,
  mse DOUBLE PRECISION NOT NULL,
  computed_by TEXT,
  runtime_ms INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (prog_hash, a_value)
);

CREATE INDEX IF NOT EXISTS idx_exec_cache_proghash ON exec_cache(prog_hash);
