# Contributing

Thank you for your interest in contributing to this project!

## Reporting Issues

- Search [existing issues](../../issues) before opening a new one.
- Include as much detail as possible: OS, Python version, error messages, and steps to reproduce.

## Submitting Pull Requests

1. Fork the repository and create a branch from `main`.
2. Follow the setup instructions in the [README](README.md).
3. Make your changes — keep commits focused and atomic.
4. Ensure the backend starts without errors (`uvicorn app.main:app --reload`) and the frontend builds (`npm run build`).
5. Open a pull request with a clear description of what you changed and why.

## Development Setup

```bash
# Backend
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env  # fill in your API keys

# Frontend
cd frontend
npm install
npm run dev
```

## Code Style

- Python: follow [PEP 8](https://peps.python.org/pep-0008/). Use type hints where practical.
- TypeScript: follow the existing ESLint/Prettier config.
- Keep pull requests small and focused on a single concern.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
