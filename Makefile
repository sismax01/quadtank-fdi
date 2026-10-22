.PHONY: install test benchmark clean

install:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

pytest:
	pytest

benchmark:
	PYTHONPATH=src python3 -m quadtank_fdi.cli --output-dir outputs

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache outputs/
	find . -name "__pycache__" -type d -exec rm -rf {} +
