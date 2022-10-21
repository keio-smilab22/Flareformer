LINTING_DIRS    := src
TEST_DIRS       := tests
TEST_TARGET_DIR := src

.PHONY: black-check
black-check:
	poetry run black --check $(LINTING_DIRS)

.PHONY: black
black:
	poetry run black $(LINTING_DIRS)
	poetry run black $(TEST_DIRS)

.PHONY: flake8
flake8:
	poetry run flake8 $(LINTING_DIRS)
	poetry run flake8 $(TEST_DIRS)

.PHONY: isort-check
isort-check:
	poetry run isort --check-only $(LINTING_DIRS)
	poetry run isort --check-only $(TEST_DIRS)

.PHONY: isort
isort:
	poetry run isort $(LINTING_DIRS)

.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: test
test:
	poetry run pytest $(TEST_DIRS) --cov=$(TEST_TARGET_DIR) --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: lint-min
lint-min:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8

.PHONY: test-all
test-all:
	$(MAKE) black
	$(MAKE) flake8
	$(MAKE) isort
	$(MAKE) mypy
	$(MAKE) test
