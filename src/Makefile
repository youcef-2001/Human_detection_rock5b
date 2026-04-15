.PHONY: help install test lint format clean run debug

help:
	@echo "Human Detection API - Development Commands"
	@echo "==========================================="
	@echo "make install       - Install dependencies"
	@echo "make test          - Run all tests"
	@echo "make test-cov      - Run tests with coverage"
	@echo "make lint          - Run linters (flake8, isort)"
	@echo "make format        - Format code (black, isort)"
	@echo "make clean         - Remove build artifacts and cache"
	@echo "make run           - Run development server"
	@echo "make run-prod      - Run production server"
	@echo "make debug         - Run with debug mode enabled"

install:
	pip install -r ../requirements.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=app --cov-report=html

lint:
	flake8 app/ tests/ run.py
	isort app/ tests/ run.py --check-only

format:
	black app/ tests/ run.py
	isort app/ tests/ run.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

run:
	python run.py --debug

run-prod:
	python run.py

debug:
	FLASK_DEBUG=true python run.py

# Fast format check without modifying
check-format:
	black --check app/ tests/ run.py

# Type checking
type-check:
	mypy app/ --ignore-missing-imports

# Complete pre-commit checks
pre-commit: lint type-check test
	@echo "All pre-commit checks passed!"
