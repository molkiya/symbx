# SymbX

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE_AGPL)
[![Commercial License Available](https://img.shields.io/badge/Commercial%20License-Available-green.svg)](LICENSE_COMMERCIAL)

**Language**: [🇷🇺 Русский](README.md) | [🇬🇧 English](README_EN.md)

SymbX is a symbolic regression system using Reinforcement Learning (RL). The project enables finding mathematical programs that transform input values into target values using a set of discrete rules (e.g., addition and multiplication).

**License**: Dual Licensing (AGPL-3.0 / Commercial) | **Author**: Marat Kiiamov (2025)

## 🎯 Key Features

- **Symbolic Regression**: Finding programs that transform input values into targets
- **Reinforcement Learning**: Using RL to train a policy for rule selection
- **Program Database**: Storage and caching of programs in PostgreSQL
- **REST API**: FastAPI for system interaction
- **Data Storage**: Integration with MinIO (S3-compatible storage)

## 📁 Project Structure

```
SymbX/
├── main.py                 # Main RL model training module
├── symbx_rules.py          # Rule definitions and program execution
├── symbx_db.py             # PostgreSQL and S3 database operations
├── app/
│   └── main_api.py         # FastAPI application
├── db/
│   └── schema.sql          # Database schema
├── demo/
│   └── demo_register_and_run.py  # Demo script
├── docs/
│   └── run.md              # Setup instructions
├── docker-compose.yml      # Infrastructure configuration
├── requirements.txt        # Python dependencies
└── Makefile               # Project management commands
```

## 🚀 Quick Start

### Requirements

- Python 3.12+
- Docker and Docker Compose
- PostgreSQL 15 (runs via Docker)
- MinIO (runs via Docker)

### Installation

1. **Clone the repository** (if not already done)

2. **Create a virtual environment and install dependencies:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Start the infrastructure:**

```bash
make infra-up
# or
docker compose up -d
```

This will start:
- **PostgreSQL** on port `5433`
- **MinIO** on ports `9000` (API) and `9001` (web console)

4. **Create a bucket in MinIO:**

   - Open [http://localhost:9001](http://localhost:9001)
   - Log in with credentials:
     - **Username:** `symbx`
     - **Password:** `replicax12345`
   - Create a bucket named `symbx`

5. **Initialize rules in the database:**

```bash
make seed
# or
python3 -c "from symbx_db import SymbXDB; SymbXDB().bootstrap_rules(); print('Seeded rules')"
```

## 📖 Usage

### Training the Model

Run RL model training:

```bash
python main.py
```

The model will train on synthetic tasks using rule combinations (addition and multiplication).

### Starting the API Server

```bash
make api
# or
uvicorn app.main_api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### Demonstration

```bash
make curl-demo
```

Or run the Python script:

```bash
python demo/demo_register_and_run.py
```

## 🔧 API Endpoints

### Health Check
```bash
GET /health
```

### Bootstrap Rules
```bash
POST /rules/bootstrap
```

### Upsert Program
```bash
POST /programs
Content-Type: application/json

{
  "prog_names": ["*3.0", "+2.0", "*2.0"],
  "complexity": 3
}
```

### Execute Program
```bash
POST /execute
Content-Type: application/json

{
  "prog_names": ["*3.0", "+2.0", "*2.0"],
  "a_value": 1.5,
  "computed_by": "cpu"
}
```

### Get Program
```bash
GET /program/{prog_hash}
```

### Log Episode
```bash
POST /episodes
Content-Type: application/json

{
  "experiment_id": 1,
  "prog_names": ["*3.0", "+2.0"],
  "x_pred": 12.0,
  "mse": 0.0,
  "reward": 1.0,
  "steps_count": 2,
  "complexity": 2
}
```

## 🛠️ Makefile Commands

- `make infra-up` - Start infrastructure (PostgreSQL, MinIO)
- `make infra-down` - Stop infrastructure and remove volumes
- `make api` - Start API server
- `make psql` - Connect to PostgreSQL
- `make seed` - Initialize rules in the database
- `make curl-demo` - Run demonstration requests
- `make pdf` - Compile LaTeX document to PDF

## 🧪 Rules

The system uses discrete rules to build programs:

- **Addition**: `+1.0`, `+2.0`, `+3.0`
- **Multiplication**: `*2.0`, `*3.0`

Rules can be extended in the `symbx_rules.py` file.

## 🗄️ Database

The database stores:

- **Rules** (`rule`, `rule_version`) - rule definitions and their versions
- **Programs** (`program`, `program_step`) - saved programs
- **Tasks** (`task`, `dataset`) - input data for training
- **Episodes** (`episode`) - program execution results
- **Execution Cache** (`exec_cache`) - cached results

The database schema is in `db/schema.sql`.

## 🔍 Architecture

1. **Training** (`main.py`):
   - Synthetic task generation
   - Policy training using RL (REINFORCE)
   - Imitation learning for pretraining

2. **Execution** (`symbx_rules.py`):
   - Rule definitions
   - Program execution
   - Canonical form and hashing

3. **Storage** (`symbx_db.py`):
   - PostgreSQL operations
   - MinIO (S3) integration
   - Execution result caching

4. **API** (`app/main_api.py`):
   - REST API for system interaction
   - Program and episode management

## 📝 Configuration

Settings can be changed via environment variables or in code:

- `PG_DSN` - PostgreSQL connection string
- `S3_ENDPOINT` - MinIO address
- `S3_ACCESS_KEY` - S3 access key
- `S3_SECRET_KEY` - S3 secret key
- `S3_BUCKET` - S3 bucket name

## 📄 License

The project is distributed under a **dual license**: AGPL-3.0 for open source use and a commercial license for proprietary use.

### Choose the appropriate license:

#### 🔓 **AGPL-3.0** (GNU Affero General Public License v3.0) — for open source

**Use if**: You are willing to open source your product.

✅ **Permitted**:
- Use in open source projects
- Modification and distribution
- Commercial use (with open source requirement)

⚠️ **Requirements**:
- When using the code (including via API), you must open source your product
- All modifications must be open source
- Preserve copyright notices

📄 See [LICENSE_AGPL](LICENSE_AGPL) file for the full license text.

#### 💼 **Commercial License** — for proprietary use

**Use if**: You want to use the code in closed commercial products without open sourcing.

✅ **Permitted**:
- Use in proprietary products
- Closed source code
- SaaS (Software as a Service) without open sourcing the service code
- Modification for internal use

⚠️ **Requirements**:
- Must purchase a commercial license
- Contact the author to discuss terms and pricing

📄 See [LICENSE_COMMERCIAL](LICENSE_COMMERCIAL) file for the full license text.

📧 **To obtain a commercial license**: Contact the author to discuss usage terms.

### 🔑 Copyright

**Copyright remains with the author** (Marat Kiiamov). Licenses grant permission to use but do not transfer ownership rights.

## 🤝 Contributing

The project is under active development. Suggestions and improvements are welcome!

## 📚 Additional Documentation

- **Mathematics and Algorithms** — mathematical problem formulation, training algorithms, and pseudocode (in development)
- **LaTeX Documentation** — professional PDF version with formulas (in development)

> **Note**: Detailed mathematical documentation is in development. Basic project information is provided in this README.

