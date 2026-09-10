.PHONY: test smoke clean-smoke

test:
	PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider

smoke:
	PYTHONDONTWRITEBYTECODE=1 python scripts/smoke.py

clean-smoke:
	rm -rf results/smoke
