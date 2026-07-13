.PHONY: setup generate train run test lint freeze

setup:
	python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

generate:
	.venv/bin/python scripts/generate_dataset.py

train:
	.venv/bin/python scripts/train_model.py

run:
	.venv/bin/uvicorn app.main:app --reload

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m py_compile app/main.py app/schemas.py app/services/features.py app/services/model_service.py app/services/training.py scripts/train_model.py

freeze:
	.venv/bin/python -m pip freeze > requirements.lock.txt
