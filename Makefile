lint:
	poetry run black .
	poetry run isort .

cli:
	poetry run python cli/main.py

api:
	poetry run uvicorn api.main:app

dash:
	poetry run streamlit run dashboard/main.py

.PHONY: cli api dash
