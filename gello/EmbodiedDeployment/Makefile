.PHONY: help install install-dev clean format lint test pre-commit git-config docs-serve docs-build docs-check pypi-serve
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env-install: ## Install package dependencies
	pip install -r envs/pip/req_install.txt

env-install-dev: ## Install development dependencies
	pip install -r envs/pip/req_develop.txt

wheel: ## Build wheel package
	pip install -r envs/pip/req_install.txt
	python setup.py bdist_wheel

clean: ## Clean temporary files and caches
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
	rm -rf site/

format: ## Format code with black and isort
	black .
	isort .

new-branch: ## Create a new branch
	git checkout -b $(branch_name)
	git push -u origin $(branch_name)

lint: ## Run linting checks
	flake8 .
	pydocstyle .

test: ## Run tests
	pytest unit_test

example: ## Run examples for testing
	bash playground/xdeploy/example.sh

pre-commit: ## Setup pre-commit hooks
	pip install -r envs/pip/req_pre_commit.txt
	cp envs/commit/commit-msg .git/hooks
	chmod +x .git/hooks/commit-msg
	pre-commit install


git-config: ## Configure git user name and email (use GIT_USER_NAME and GIT_USER_EMAIL env vars)
	@if [ -z "$(GIT_USER_NAME)" ]; then \
		echo "Error: GIT_USER_NAME is not set"; \
		echo "Usage: make git_config GIT_USER_NAME=your_name GIT_USER_EMAIL=your_email@example.com"; \
		exit 1; \
	fi
	@if [ -z "$(GIT_USER_EMAIL)" ]; then \
		echo "Error: GIT_USER_EMAIL is not set"; \
		echo "Usage: make git_config GIT_USER_NAME=your_name GIT_USER_EMAIL=your_email@example.com"; \
		exit 1; \
	fi
	@git config --global user.name "$(GIT_USER_NAME)"
	@git config --global user.email "$(GIT_USER_EMAIL)"
	@echo "Git config updated:"
	@echo "  user.name: $(shell git config --global user.name)"
	@echo "  user.email: $(shell git config --global user.email)"

pypi-serve: ## [pypi] Start local pypi server
	pypi-server run -i 0.0.0.0 -p 12345 .

docs-serve: ## [docs] Start local documentation server (with hot reload)
	mkdocs serve --dev-addr=0.0.0.0:54321

docs-build: ## [docs] Build documentation to static site
	mkdocs build

docs-check: ## [docs] Check API documentation for missing docstrings
	PYTHONPATH=. python playground/scripts/check_docs_api.py --check-methods
