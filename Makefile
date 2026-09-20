run-backend:
	uvicorn backend.main:app --reload --port 8000

run-frontend:
	streamlit run frontend/app.py

test:
	pytest -q

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
