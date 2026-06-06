# Repository Guidelines

## Project Structure & Module Organization

Glassbox is a Python 3.12 project using a `src` layout.

- `src/glassbox/`: application package.
- `src/glassbox/server/`: FastAPI app, routes, and request/response schemas.
- `src/glassbox/engine/`: model loading and inference logic.
- `src/glassbox/config.py`: environment-driven runtime configuration.
- `tests/`: pytest tests; server-specific tests live in `tests/server/`.
- `docs/project-plan.md`: architecture, milestones, configuration, and scope.

Keep HTTP concerns in `server`, inference behavior in `engine`, and shared configuration at the package root. Do not add deferred features such as batching or streaming without aligning them with the project plan.

## Build, Test, and Development Commands

The project uses `uv` for Python and package management:

```bash
uv sync                         # Create/update the local environment
uv build                        # Build wheel and source distributions
uv run --with pytest pytest     # Run the complete test suite
uv run --with pytest pytest tests/server/test_health.py
```

As dependencies are added, declare them in `pyproject.toml` rather than relying on globally installed packages. The intended local API entry point is the `glassbox` script declared in `pyproject.toml`.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations for public interfaces, and concise docstrings where behavior is not obvious. Follow standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Prefer small modules with explicit boundaries over broad utility files.

No formatter or linter is currently configured. Keep code PEP 8 compliant and avoid introducing tool-specific configuration unless it is added consistently for the repository.

## Testing Guidelines

Write tests with pytest. Name files `test_*.py` and test functions `test_<behavior>`. Mirror source ownership where useful, such as `tests/server/` for API behavior. Cover configuration defaults and overrides, endpoint status and schemas, and engine behavior without requiring a GPU unless the test is explicitly marked as hardware-dependent.

## Commit & Pull Request Guidelines

Current history uses short, sentence-style commit subjects without prefixes, for example `Wrote readme`. Keep each commit focused and describe the completed change clearly.

Pull requests should include a concise purpose, implementation notes, test commands run, and linked issues when applicable. Include example API requests/responses for endpoint changes and benchmark results for inference-performance changes.
