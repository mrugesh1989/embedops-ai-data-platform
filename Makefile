.PHONY: up down build logs ps restart clean

up:
	docker compose up --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

restart:
	docker compose down -v
	docker compose up --build

clean:
	docker compose down -v