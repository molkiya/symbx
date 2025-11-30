.PHONY: infra-up infra-down api dev psql seed curl-demo pdf

infra-up:
	docker compose up -d

infra-down:
	docker compose down -v

psql:
	PGPASSWORD=symbx psql -h 127.0.0.1 -p 5433 -U symbx -d symbx

api:
	uvicorn app.main_api:app --reload --host 0.0.0.0 --port 8000

seed:
	python3 -c "from symbx_db import SymbXDB; SymbXDB().bootstrap_rules(); print('Seeded rules')"

curl-demo:
	curl -s http://127.0.0.1:8000/health | jq
	curl -s -X POST http://127.0.0.1:8000/rules/bootstrap | jq
	curl -s -X POST http://127.0.0.1:8000/programs -H 'content-type: application/json' -d '{\"prog_names\":[\"*3.0\",\"+2.0\",\"*2.0\"],\"complexity\":3}' | jq
	curl -s -X POST http://127.0.0.1:8000/execute -H 'content-type: application/json' -d '{\"prog_names\":[\"*3.0\",\"+2.0\",\"*2.0\"],\"a_value\":1.5}' | jq

pdf:
	cd docs && xelatex -interaction=nonstopmode math.tex && xelatex -interaction=nonstopmode math.tex
