# Contributing to the Project

We warmly welcome contributions from the community! This guide will help you get started with contributing to our project.

## Ways to Contribute

There are many ways you can contribute:

1. **Reporting Bugs**
    - Open clear, detailed issues on our GitHub repository
    - Include:
        - Steps to reproduce
        - Expected behavior
        - Actual behavior
        - Your environment details

2. **Suggesting Enhancements**
    - Submit feature requests as GitHub issues
    - Provide:
        - Clear description of the proposed feature
        - Potential use cases
        - Any initial thoughts on implementation

3. **Code Contributions**
    - Fork the repository
    - Create a new branch for your feature or bugfix
    - Write clear, documented code
    - Ensure all tests pass
    - Add tests for your code, if your changes are complex
    - Submit a pull request

## Development Setup
```bash
# Clone the repository
git clone https://github.com/flixOpt/flixopt.git
cd flixopt

# It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install development dependencies
pip install -e .[dev]  # This installs the package in editable mode with development dependencies

# Create a new branch
git checkout -b feature/your-feature-name

# Run tests
pytest

# Run a linter to improve code quality
ruff check . --fix
```

# Best practices

## Coding Guidelines

- Follow PEP 8 style guidelines
- Write clear, commented code
- Include type hints
- Create or update tests for new functionality
- Ensure 100% test coverage for new code

## Branches
As we start to think flixOpt in **Releases**, we decided to introduce multiple **dev**-branches instead of only one:
Following the **Semantic Versioning** guidelines, we introduced:
- `next/patch`: This is where all pull requests for the next patch release (1.0.x) go.  
- `next/minor`: This is where all pull requests for the next minor release (1.x.0) go.  
- `next/major`: This is where all pull requests for the next major release (x.0.0) go.

- Everything else remains in `feature/...`-branches.

## Pull requests
Every feature or bugfix should be merged into one of the 3 [release branches](#branches), using **Squash and merge** or a regular **single commit**.
At some point, `next/minor` or `next/major` will get merged into `main` using a regular **Merge**  (not squash).
*This ensures that Features are kept separate, and the `next/...`branches stay in synch with ``main`.*

**Remember to update the version in `pyproject.toml`**

## Releases
As stated, we follow **Semantic Versioning**.
Right after one of the 3 [release branches](#Branches) is merged into main, a **Tag** should be added to the merge commit and pushed to the main branch. The tag has the form `v1.2.3`.
With this tag,  a release with **Release Notes** must be created. 

*This is our current best practice*
