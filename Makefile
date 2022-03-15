build:
	@docker-compose build

sh:
	@docker-compose run --rm dev

.PHONY: build sh
