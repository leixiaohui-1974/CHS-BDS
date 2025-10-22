.PHONY: help install test lint format clean docker-build docker-run docker-stop docs

# Default target
help:
	@echo "CHS-BDS Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Installation:"
	@echo "  make install        - Install package in development mode"
	@echo "  make install-dev    - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run unit tests"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run linters (flake8)"
	@echo "  make format         - Format code with black"
	@echo "  make typecheck      - Run type checking with mypy"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run container"
	@echo "  make docker-stop    - Stop container"
	@echo "  make docker-clean   - Remove container and image"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make run-all        - Run all monitoring modules"
	@echo "  make docs           - Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest

test-coverage:
	pytest --cov=gnss_monitoring --cov-report=html --cov-report=term

test-verbose:
	pytest -vv -s

# Code quality
lint:
	flake8 gnss_monitoring tests --max-line-length=100 --ignore=E203,W503

format:
	black gnss_monitoring tests --line-length=100

typecheck:
	mypy gnss_monitoring --ignore-missing-imports

# Docker
docker-build:
	docker build -t chs-bds:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker rmi chs-bds:latest

docker-logs:
	docker-compose logs -f

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -f *.png *.log

run-all:
	python -m gnss_monitoring.main --all

run-module:
	@echo "Usage: make run-module MODULE=<gnss_ir|deformation|pwv|rainfall>"
	python -m gnss_monitoring.main --module $(MODULE)

docs:
	@echo "Documentation generation not yet implemented"
	@echo "Visit: https://github.com/leixiaohui-1974/CHS-BDS"

# Create necessary directories
setup-dirs:
	mkdir -p data output logs

# Quick start
quickstart: install setup-dirs
	@echo "CHS-BDS setup complete!"
	@echo "Run 'make run-all' to test all modules"
