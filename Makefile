.PHONY: install test lint clean notebook

install:
	pip install -e ".[serve,dev]"

test:
	pytest tests/ -v

lint:
	ruff check portugal_housing/ deployment/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

notebook:
	jupyter lab notebooks/portugal_housing_prediction.ipynb
