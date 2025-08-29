.PHONY: infra-up infra-down api dev psql seed curl-demo

infra-up:
\tdocker compose up -d

infra-down:
\tdocker compose down -v

psql:
\tPGPASSWORD=symbx psql -h 127.0.0.1 -p 5433 -U symbx -d symbx

api:
\tuvicorn app.main_api:app --reload --host 0.0.0.0 --port 8000

seed:
\tpython3 -c "from replicax_db import ReplicaXDB; ReplicaXDB().bootstrap_rules(); print('Seeded rules')"

curl-demo:
\tcurl -s http://127.0.0.1:8000/health | jq
\tcurl -s -X POST http://127.0.0.1:8000/rules/bootstrap | jq
\tcurl -s -X POST http://127.0.0.1:8000/programs -H 'content-type: application/json' -d '{\"prog_names\":[\"*3.0\",\"+2.0\",\"*2.0\"],\"complexity\":3}' | jq
\tcurl -s -X POST http://127.0.0.1:8000/execute -H 'content-type: application/json' -d '{\"prog_names\":[\"*3.0\",\"+2.0\",\"*2.0\"],\"a_value\":1.5}' | jq
