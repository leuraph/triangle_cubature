# Python Package Template
This repo serves as a template for python packages.

## (Unit) Tests
To run auto tests, you do
```sh
python -m unittest discover tests/auto/
```

## Development Tools
Any development tools, not need in the final package but
needed in tests are located in dedicated directory `src/dev_tools/`.
This package is explicitly excluded in the `pyproject.toml` file
but, nevertheles, still included in editable installs.

## editable install
```sh
$ cd your-python-project
$ python -m venv .venv
# Activate your environment with:
#      `source .venv/bin/activate` on Unix/macOS
# or   `.venv\Scripts\activate` on Windows

$ pip install --editable .

# Now you have access to your package
# as if it was installed in .venv
$ python -c "import your_python_project"
```
As a further reference, see
[here](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).