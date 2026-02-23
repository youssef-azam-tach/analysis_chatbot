# AI Analysis Platform (Monorepo)

End-to-end AI analytics platform with three main runtime services:

- Core Backend (authentication, users, workspaces, orchestration)
- AI Engine (analysis, cleaning, schema, visualization, chat, reports)
- Frontend (React + Vite SPA)

This repository also contains the shared AI modules used by the AI Engine.

## Repository Structure

```text
analysis-everything/
├── AI-partion/                     # Shared AI modules (analysis/models/pipelines/ui)
├── Backend_apis/
│   ├── fastapi-backend/            # Service 1: Core backend API
│   └── fastapi-api/                # Service 2: AI engine API
├── frontend/                       # Service 3: React frontend
├── docker-compose.yml              # Full stack runtime
├── API_DOCUMENTATION.md            # API reference
└── README.md
```

## Architecture

- PostgreSQL for core backend persistence
- Core backend connects to AI engine for advanced analysis workflows
- AI engine mounts [AI-partion](AI-partion) as read-only modules directory
- Frontend proxies requests to backend and AI engine through Nginx configuration

## Main Capabilities

- Multi-file upload and dataset session management
- Schema analysis + manual relationships
- Data quality checks and cleaning workflows
- Strategic AI analysis with chart recommendations
- KPI and visualization builders
- Dashboard builder (multi-widget)
- Persistent AI chat with pins
- Report export (PDF/PPTX)
- Quick single-file analysis page outside the main pipeline flow

## Prerequisites

- Docker + Docker Compose
- GNU Make
- (Recommended) Node.js 20+ for local frontend development
- (Recommended) Python 3.11+ for local backend development
- Ollama running on host (default expected at `127.0.0.1:11434`)

## Environment Configuration

Create a root `.env` file (next to `docker-compose.yml`) with at least these keys:

```env
COMPOSE_PROJECT_NAME=ai-analysis
IMAGE_VERSION=1.0.1

APP_NAME_BACKEND=ai-analysis-backend
APP_NAME_ENGINE=ai-analysis-engine
APP_NAME_FRONTEND=ai-analysis-frontend

POSTGRES_DB=ai_analysis
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=4809

BACKEND_PORT=9867
ENGINE_PORT=3490
FRONTEND_PORT=8081

OLLAMA_HOST=http://host.docker.internal:11435
```

Adjust values to your environment as needed.

## Run with Docker (Recommended)

1) Build images

```bash
cd Backend_apis/fastapi-backend && make build
cd ../fastapi-api && make build
cd ../../frontend && make build
cd ..
```

2) Start full stack

```bash
docker compose up -d
```

3) Check status and logs

```bash
docker compose ps
docker compose logs -f frontend
docker compose logs -f backend
docker compose logs -f ai-engine
```

4) Stop stack

```bash
docker compose down
```

To remove volumes too:

```bash
docker compose down -v
```

## Local Development (Without Docker)

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

### AI Engine (Service 2)

```bash
cd Backend_apis/fastapi-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 3490
```

### Core Backend (Service 1)

```bash
cd Backend_apis/fastapi-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 9867
```

## API Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for endpoint details and payload examples.

## Important Notes

- Keep secrets only in `.env` and do not commit real credentials.
- The root [.gitignore](.gitignore) is configured for Python, Node, logs, env files, and generated runtime artifacts.
- If Docker build fails in frontend, run `npm run build` inside [frontend](frontend) to inspect TypeScript errors directly.

## Troubleshooting

- Ollama errors:
  - Ensure host Ollama is running.
  - Verify `OLLAMA_HOST` matches your setup.
- Port conflicts:
  - Change `BACKEND_PORT`, `ENGINE_PORT`, `FRONTEND_PORT`, and `POSTGRES_PORT` in `.env`.
- Missing images:
  - Re-run the three `make build` commands before `docker compose up -d`.