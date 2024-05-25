'''
Run this script to run all tests with pytest
Alternatively, use 'python -m unittest discover -s tests' in the command line (using unittest)
or run testmodules individually
'''
import pytest
pytest.main(['-v', '--disable-warnings'])