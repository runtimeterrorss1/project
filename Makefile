install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install --force-reinstall -v "fsspec==2022.11.0"