# Testing

To run the test methods, run the following from this folder:

```
python -m unittest discover
```

This will show the outcome of all tests.


To assess test coverage of the codebase, use:

```
py.test --cov gryds --cov-report=test
```

to show in terminal or 

```
py.test --cov gryds --cov-report=html
```

to create an interactive html-based test coverage report.
