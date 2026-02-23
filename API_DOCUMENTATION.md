# Verified AI Analytics Platform — API Documentation

> **Decision-Grade AI Analytics Engine with Trust Layer**
> Every insight is backed by evidence. No hallucinated claims. No unverified signals.

---

## Architecture Overview

| Component | Host | Port | Image |
|-----------|------|------|-------|
| **Backend API** (Auth + Core) | `10.100.102.6` | `9867` | `ai-analysis-backend:latest` |
| **AI Engine** (Analysis + Trust) | `10.100.102.6` | `3490` | `ai-analysis-engine:latest` |
| **PostgreSQL 16** | `10.100.102.6` | `4809` | `postgres:16-alpine` |

**Base URLs:**
- Backend: `http://10.100.102.6:9867/api/v1`
- AI Engine: `http://10.100.102.6:3490/api/v1`

---

## Design Principles

1. **No Organizations, No Roles, No Teams** — Simple user authentication. Every user owns their own workspaces and data.
2. **Trust Layer** — Every AI insight must be backed by data evidence. The system includes Insight Evidence Mapping, Contradiction Detection, and Decision Relevance Evaluation.
3. **Multiple Data Sources** — Support for Excel/CSV file uploads AND live database connections (PostgreSQL, MySQL, etc.).
4. **Dynamic Page Visibility** — Pages shown to users depend on (a) user preferences and (b) data source type (database workspaces hide cleaning/upload pages).
5. **Chatbot Validation Mode** — Strict mode ensures the chatbot never returns an insight without evidence blocks.
6. **Evidence Audit in Reports** — Every generated report includes an evidence audit section showing verification status of all insights.

---

## Authentication

Simple JWT-based auth. No roles, no RBAC.

### `POST /api/v1/auth/register`

Register a new user.

```json
{
  "email": "user@example.com",
  "full_name": "Jane Doe",
  "password": "SecurePass123"
}
```

**Response** `201`:
```json
{
  "success": true,
  "data": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "Jane Doe",
    "is_active": true,
    "enabled_pages": ["dashboard", "data_upload", "data_cleaning", ...],
    "created_at": "2025-01-01T00:00:00",
    "updated_at": "2025-01-01T00:00:00"
  },
  "message": "User registered successfully"
}
```

### `POST /api/v1/auth/login`

```json
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response** `200`:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

### `POST /api/v1/auth/refresh`

```json
{ "refresh_token": "eyJ..." }
```

### `GET /api/v1/auth/me`

Returns current user profile. **Requires:** `Authorization: Bearer <token>`

### `POST /api/v1/auth/change-password`

```json
{
  "current_password": "OldPass",
  "new_password": "NewPass123"
}
```

---

## Users

All endpoints require `Authorization: Bearer <token>`.

### `GET /api/v1/users/`
List users. Query params: `page`, `page_size`.

### `GET /api/v1/users/{user_id}`
Get user by ID.

### `PUT /api/v1/users/{user_id}`
Update own profile (full_name, avatar_url, is_active).

### `PUT /api/v1/users/{user_id}/pages`
Update page visibility preferences.

```json
{
  "enabled_pages": ["dashboard", "eda", "kpis", "chatbot", "trust_layer"]
}
```

### `DELETE /api/v1/users/{user_id}`
Delete own account.

---

## Workspaces

Workspaces are owned by users. Each workspace has a `data_source_type` ("file" or "database") that controls which pages are available.

### `POST /api/v1/workspaces/`

```json
{
  "name": "Q4 Sales Analysis",
  "description": "Quarterly sales data analysis",
  "data_source_type": "file",
  "enabled_pages": ["dashboard", "eda", "kpis"]
}
```

### `GET /api/v1/workspaces/`
List all workspaces owned by current user.

### `GET /api/v1/workspaces/{ws_id}`
Get workspace by ID (must be owner).

### `PUT /api/v1/workspaces/{ws_id}`
Update workspace.

### `DELETE /api/v1/workspaces/{ws_id}`
Delete workspace.

---

## Data Sources

Support file uploads (Excel/CSV) and live database connections.

### `POST /api/v1/data-sources/`

**File data source:**
```json
{
  "workspace_id": 1,
  "source_type": "file",
  "file_path": "/uploads/sales_q4.xlsx",
  "file_type": "xlsx",
  "file_size_bytes": 524288
}
```

**Database data source:**
```json
{
  "workspace_id": 1,
  "source_type": "database",
  "db_type": "postgresql",
  "db_host": "10.100.102.6",
  "db_port": 5432,
  "db_name": "production_db",
  "db_user": "analyst",
  "db_password": "secret",
  "db_schema": "public",
  "db_ssl": true
}
```

### `POST /api/v1/data-sources/test-connection`

Test a database connection before saving.

```json
{
  "db_type": "postgresql",
  "db_host": "10.100.102.6",
  "db_port": 5432,
  "db_name": "production_db",
  "db_user": "analyst",
  "db_password": "secret"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "success": true,
    "message": "Connected to postgresql://10.100.102.6:5432/production_db",
    "tables": ["orders", "customers", "products"],
    "schemas": ["public", "analytics"]
  }
}
```

### `GET /api/v1/data-sources/workspace/{workspace_id}`
List all data sources in a workspace.

### `GET /api/v1/data-sources/{ds_id}`
Get data source by ID.

### `PUT /api/v1/data-sources/{ds_id}`
Update data source connection details.

### `DELETE /api/v1/data-sources/{ds_id}`
Delete data source.

---

## Page Configuration

Dynamic page visibility based on user preferences + workspace data source type.

### Available Pages

| Slug | Description |
|------|-------------|
| `dashboard` | Main dashboard |
| `data_upload` | File upload (hidden for DB workspaces) |
| `data_cleaning` | Data cleaning tools (hidden for DB workspaces) |
| `eda` | Exploratory Data Analysis |
| `advanced_analysis` | Advanced analytics |
| `kpis` | KPI intelligence |
| `visualization` | Chart builder |
| `reports` | Report generation |
| `chatbot` | AI chatbot |
| `rag` | RAG Q&A |
| `schema_analysis` | Multi-dataset schema analysis |
| `trust_layer` | Trust Layer / Evidence audit |
| `contradiction_detection` | Contradiction detection |
| `decision_relevance` | Decision relevance evaluator |

### `GET /api/v1/pages/available`
Get all available page slugs.

### `GET /api/v1/pages/visible/{workspace_id}`
Get pages visible for the current user in the given workspace.

**Logic:**
1. Start with user's `enabled_pages` preference
2. Intersect with workspace's `enabled_pages` override (if set)
3. If `data_source_type == "database"`, remove `data_cleaning` and `data_upload`

**Response:**
```json
{
  "success": true,
  "data": {
    "workspace_id": 1,
    "data_source_type": "database",
    "visible_pages": ["dashboard", "eda", "kpis", "chatbot", "trust_layer"]
  }
}
```

---

## Trust Layer (Backend — Service 1)

Persists verified insights and contradictions to the database.

### Verified Insights

#### `POST /api/v1/trust/insights`

Save a verified insight with evidence.

```json
{
  "workspace_id": 1,
  "insight_text": "Revenue increased 15% QoQ driven by enterprise segment",
  "insight_category": "trend",
  "evidence": [
    {
      "source": "column_stats",
      "description": "Revenue column mean increased from $50K to $57.5K",
      "data_reference": "revenue",
      "confidence": 0.92,
      "raw_data": {"mean_q3": 50000, "mean_q4": 57500}
    }
  ],
  "confidence_score": 0.92,
  "decision_area": "finance",
  "recommended_action": "Increase enterprise sales resources",
  "risk_if_wrong": "Over-investment in enterprise at expense of SMB",
  "signal_strength": "strong"
}
```

#### `GET /api/v1/trust/insights/workspace/{workspace_id}`
Get all insights for a workspace. Optional query: `?category=trend`

#### `GET /api/v1/trust/insights/workspace/{workspace_id}/verified`
Get only fully verified insights.

#### `GET /api/v1/trust/insights/decision/{workspace_id}?decision_area=pricing`
Get insights relevant to a specific decision area.

### Contradictions

#### `POST /api/v1/trust/contradictions`

Log a detected contradiction.

```json
{
  "workspace_id": 1,
  "signal_a": "Revenue is growing 15%",
  "signal_b": "Revenue declined 5% month-over-month",
  "conflict_summary": "Opposing signals for revenue trend",
  "impact_on_decision": "Cannot determine revenue trajectory",
  "severity_score": 0.85,
  "evidence_a": {"source": "quarterly_trend"},
  "evidence_b": {"source": "monthly_trend"}
}
```

#### `GET /api/v1/trust/contradictions/workspace/{workspace_id}`
Get contradictions. Optional: `?unresolved_only=true`

#### `PUT /api/v1/trust/contradictions/{id}/resolve`
Mark as resolved.
```json
{ "resolution_note": "Monthly dip was seasonal; quarterly trend is accurate." }
```

---

## Trust Layer (AI Engine — Service 2)

Real-time trust engines that verify insights against live data.

### `POST /api/v1/trust/verify-insight`

Verify a single insight against the dataset.

```json
{
  "session_id": "abc123",
  "insight_text": "Sales are increasing over time",
  "model": "qwen2.5:7b"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "insight_text": "Sales are increasing over time",
    "verification_status": "verified",
    "confidence_score": 0.87,
    "evidence": [
      {
        "source": "trend_analysis",
        "description": "'sales' shows increasing trend (85% directional consistency)",
        "data_reference": "sales",
        "confidence": 0.85,
        "raw_data": {"trend": "increasing", "positive_ratio": 0.85}
      },
      {
        "source": "column_stats",
        "description": "Statistical profile for 'sales'",
        "data_reference": "sales",
        "confidence": 0.9,
        "raw_data": {"mean": 15230.5, "std": 4521.3, "min": 2100, "max": 45000}
      }
    ],
    "signal_strength": "strong"
  }
}
```

### `POST /api/v1/trust/verify-analysis`

Verify all insights from analysis results.

```json
{
  "session_id": "abc123",
  "analysis_results": {
    "insights": [
      {"insight": "Revenue growing", "category": "trend"},
      {"insight": "High outlier count in expenses", "category": "anomaly"}
    ]
  }
}
```

### `POST /api/v1/trust/detect-contradictions`

Check insights for conflicting signals.

```json
{
  "insights": [
    {"insight_text": "Revenue grew 15%", "evidence": [...], "confidence_score": 0.9},
    {"insight_text": "Revenue declined 5%", "evidence": [...], "confidence_score": 0.8}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "has_contradictions": true,
    "contradictions": [
      {
        "signal_a": "Revenue grew 15%",
        "signal_b": "Revenue declined 5%",
        "conflict_summary": "Opposing signals for revenue",
        "severity_score": 0.85,
        "shared_references": ["revenue"]
      }
    ],
    "summary": "Found 1 contradiction(s) across 2 insights."
  }
}
```

### `POST /api/v1/trust/decision-relevance`

Evaluate insights by relevance to a decision area.

```json
{
  "decision_area": "pricing strategy",
  "insights": [...]
}
```

---

## Chatbot (AI Engine)

### `POST /api/v1/chat/message`

```json
{
  "session_id": "abc123",
  "message": "What are the top revenue drivers?",
  "chatbot_type": "hybrid",
  "model": "qwen2.5:7b",
  "temperature": 0.7,
  "strict": true
}
```

**When `strict: true`**, the response includes a `trust_layer` block:

```json
{
  "success": true,
  "data": {
    "answer": "The top revenue drivers are...",
    "graphs": [],
    "trust_layer": {
      "verified": true,
      "verification_status": "verified",
      "confidence_score": 0.87,
      "signal_strength": "strong",
      "evidence": [...],
      "evidence_count": 3
    }
  }
}
```

If unverified:
```json
{
  "trust_layer": {
    "verified": false,
    "verification_status": "unverified",
    "confidence_score": 0.15,
    "signal_strength": "weak",
    "evidence": [],
    "evidence_count": 0,
    "warning": "This response could not be verified against the data. Treat as unconfirmed until evidence is available."
  }
}
```

**Chatbot types:** `hybrid`, `enhanced`, `llm`, `pandas_agent`

### `GET /api/v1/chat/history?session_id=abc123`
Get chat history.

### `POST /api/v1/chat/clear?session_id=abc123`
Clear chat history.

---

## File Management (AI Engine)

### `POST /api/v1/files/upload`
Upload Excel/CSV file. Returns `session_id`, detected sheets, column types.

### `POST /api/v1/files/set-active`
Set active dataset (sheet) within session.

### `GET /api/v1/files/datasets?session_id=abc123`
List all datasets in session.

### `POST /api/v1/files/combine`
Merge/concat multiple datasets.

### `GET /api/v1/files/preview?session_id=abc123&rows=50`
Preview first N rows.

### `GET /api/v1/files/download?session_id=abc123`
Download cleaned/processed file.

---

## Analysis (AI Engine)

### `POST /api/v1/analysis/eda`
Run EDA. Returns statistics, distributions, correlations.

### `POST /api/v1/analysis/data-quality`
Comprehensive data quality assessment.

### `POST /api/v1/analysis/business-intelligence`
AI-powered business insights.

### `POST /api/v1/analysis/advanced`
Advanced analysis with custom goals.

### `POST /api/v1/analysis/automatic`
Full automatic analysis pipeline.

### `POST /api/v1/analysis/schema`
Multi-dataset schema analysis with relationship detection.

All request bodies:
```json
{ "session_id": "abc123", "model": "qwen2.5:7b" }
```

---

## Data Cleaning (AI Engine)

> **Note:** These endpoints are hidden in database-sourced workspaces.

### `POST /api/v1/cleaning/missing` — Handle missing values
### `POST /api/v1/cleaning/outliers` — Remove outliers
### `POST /api/v1/cleaning/duplicates` — Remove duplicates
### `POST /api/v1/cleaning/encode` — Encode categoricals
### `POST /api/v1/cleaning/scale` — Scale/normalize
### `POST /api/v1/cleaning/custom-column` — Create computed column
### `POST /api/v1/cleaning/pivot` — Pivot table
### `POST /api/v1/cleaning/unpivot` — Unpivot
### `POST /api/v1/cleaning/change-type` — Change column type
### `POST /api/v1/cleaning/split-column` — Split column by delimiter
### `POST /api/v1/cleaning/merge` — Merge datasets
### `POST /api/v1/cleaning/append` — Append datasets
### `POST /api/v1/cleaning/undo` — Undo last operation
### `GET /api/v1/cleaning/history?session_id=abc123` — Operation history

---

## KPIs (AI Engine)

### `POST /api/v1/kpis/generate`
Auto-generate KPIs from data.
```json
{ "session_id": "abc123", "max_kpis": 20 }
```

### `POST /api/v1/kpis/single`
Calculate a single KPI.
```json
{ "session_id": "abc123", "column": "revenue", "aggregation": "sum" }
```

### `POST /api/v1/kpis/validate`
Validate KPI parameters.

---

## Visualization (AI Engine)

### `POST /api/v1/visualization/create`
Create a chart.
```json
{
  "session_id": "abc123",
  "chart_type": "bar",
  "params": { "x": "category", "y": "revenue", "color": "region" }
}
```

---

## Reports (AI Engine)

### `POST /api/v1/reports/generate` (Background Task)

```json
{
  "session_id": "abc123",
  "report_type": "pdf",
  "include_evidence_audit": true
}
```

Reports now include an **Evidence Audit** section:
- Total insights verified
- Breakdown: verified / partial / weak / unverified
- Contradictions found
- Detailed evidence trail for each insight

---

## Dashboards (AI Engine)

### `POST /api/v1/dashboard/build`
AI-generated dashboard.
```json
{ "session_id": "abc123", "goals": {"focus": "sales performance"} }
```

---

## RAG (AI Engine)

### `POST /api/v1/rag/setup`
Index dataset for RAG queries.
```json
{ "session_id": "abc123" }
```

### `POST /api/v1/rag/query`
Query with RAG-augmented answers.
```json
{ "session_id": "abc123", "query": "What drives revenue?", "model": "qwen2.5:7b" }
```

---

## WebSocket — Streaming Chat

### `WS /api/v1/ws/chat`

Connect:
```
ws://10.100.102.6:3490/api/v1/ws/chat?token=<jwt>&session_id=abc123
```

Send:
```json
{ "message": "Analyze revenue trends", "model": "qwen2.5:7b", "temperature": 0.7 }
```

Receive streamed tokens:
```json
{ "type": "token", "content": "Based on" }
{ "type": "token", "content": " the data..." }
{ "type": "done", "content": "" }
```

---

## Background Tasks

### `GET /api/v1/tasks/{task_id}`
Check task status.

**Response:**
```json
{
  "task_id": "uuid",
  "task_type": "report_generation",
  "status": "completed",
  "progress": 1.0,
  "result": { "file_path": "/reports/report.pdf" }
}
```

---

## Datasets (Backend)

### `POST /api/v1/datasets/`
Register dataset metadata.

### `GET /api/v1/datasets/workspace/{workspace_id}`
List datasets in workspace.

### `GET /api/v1/datasets/{id}`
Get dataset metadata.

### `DELETE /api/v1/datasets/{id}`
Delete dataset.

---

## Dashboards (Backend)

### `POST /api/v1/dashboards/`
Create dashboard.

### `GET /api/v1/dashboards/workspace/{workspace_id}`
List dashboards.

### `PUT /api/v1/dashboards/{id}`
Update dashboard.

### `DELETE /api/v1/dashboards/{id}`
Delete dashboard.

### Dashboard Items
- `POST /api/v1/dashboards/{id}/items` — Add item
- `PUT /api/v1/dashboards/{id}/items/{item_id}` — Update item
- `DELETE /api/v1/dashboards/{id}/items/{item_id}` — Remove item

---

## KPIs (Backend)

### `POST /api/v1/kpis/`
Save KPI.

### `GET /api/v1/kpis/workspace/{workspace_id}`
List saved KPIs.

### `PUT /api/v1/kpis/{id}`
Update KPI.

### `DELETE /api/v1/kpis/{id}`
Delete KPI.

---

## Reports (Backend)

### `POST /api/v1/reports/`
Save report metadata.

### `GET /api/v1/reports/workspace/{workspace_id}`
List reports.

### `GET /api/v1/reports/{id}`
Get report.

### `DELETE /api/v1/reports/{id}`
Delete report.

---

## Chat History (Backend)

### Sessions
- `POST /api/v1/chat/sessions` — Create session
- `GET /api/v1/chat/sessions?workspace_id=1` — List sessions
- `GET /api/v1/chat/sessions/{id}` — Get session with messages
- `PUT /api/v1/chat/sessions/{id}` — Update session title
- `DELETE /api/v1/chat/sessions/{id}` — Delete session

### Messages
- `POST /api/v1/chat/sessions/{id}/messages` — Add message
- `GET /api/v1/chat/sessions/{id}/messages` — Get messages

---

## Health Checks

### Backend: `GET /api/v1/health`
### AI Engine: `GET /api/v1/health`

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "service": "ai-engine",
    "version": "1.0.0"
  }
}
```

---

## Error Responses

All errors follow this format:
```json
{
  "success": false,
  "data": null,
  "message": "Error description"
}
```

| Code | Meaning |
|------|---------|
| 400 | Bad request / validation error |
| 401 | Unauthorized / invalid token |
| 403 | Forbidden / not owner |
| 404 | Resource not found |
| 422 | Validation error (Pydantic) |
| 500 | Internal server error |

---

## Environment Configuration

Single `.env` file at `Backend_apis/.env`:

```env
HOST=10.100.102.6
BACKEND_PORT=9867
ENGINE_PORT=3490
POSTGRES_PORT=4809
POSTGRES_USER=ai_admin
POSTGRES_PASSWORD=<secret>
POSTGRES_DB=ai_analysis
JWT_SECRET_KEY=<secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
OLLAMA_HOST=http://10.100.102.6:11434
OLLAMA_MODEL=qwen2.5:7b
EMBED_MODEL=nomic-embed-text
UPLOAD_DIR=/app/storage/uploads
CHROMA_DB_PATH=/app/storage/chroma
AI_MODULES_PATH=/app/ai-modules
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (async) |
| ORM | SQLAlchemy 2.0 (async + asyncpg) |
| Validation | Pydantic v2 |
| Database | PostgreSQL 16 |
| Auth | JWT (PyJWT + bcrypt) |
| LLM | Ollama (qwen2.5:7b) |
| Embeddings | nomic-embed-text |
| Vector Store | ChromaDB |
| AI Framework | LangChain |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| PDF Reports | FPDF2 |
| Containerization | Docker + Docker Compose |

---

## Phase 4 — Full Pipeline & Feature Parity Additions

> These endpoints were added to ensure every Streamlit feature has a corresponding API endpoint,
> and the full data pipeline works end-to-end for both Excel and Database data sources.

### Data Source Pipeline

When a user selects a data source:

**Excel/CSV Path:**
1. `POST /files/upload` or `POST /files/upload-multiple` — Upload files
2. `POST /analysis/schema` — Schema analysis (relationships between tables)
3. `POST /analysis/data-quality` — Data quality assessment
4. `POST /cleaning/*` — Full cleaning pipeline (missing values, outliers, duplicates, encoding, scaling, merge, append, pivot, etc.)
5. `POST /pipeline/finalize` — Mark golden dataset (**new**)
6. All analysis endpoints use the finalized dataset

**Database Path:**
1. `POST /database/query` — Load data from external DB (**new**)
2. Pages `data_cleaning`, `multi_file_loader`, `quick_excel_analysis` are hidden via `/pages/visible/{workspace_id}`
3. Analysis + visualization + chatbot endpoints work immediately

---

### Strategic Analysis Endpoints (AI Engine)

#### `POST /api/v1/strategic/goals`

Save business goals for a session. Influences strategic analysis, AI suggestions, and reports.

```json
{
  "session_id": "uuid",
  "problem": "Sales have declined by 15% in Q3",
  "objective": "Identify drivers and suggest recovery strategies",
  "target": "Regional Sales Managers"
}
```

**Response:** `{"data": {"goals": {...}}, "message": "Business goals saved"}`

#### `GET /api/v1/strategic/goals?session_id=uuid`

Get currently saved business goals.

#### `POST /api/v1/strategic/analyze`

Run full strategic analysis using business goals and ALL loaded datasets. Background task.

```json
{
  "session_id": "uuid",
  "model": "qwen2.5:7b",
  "goals": {
    "problem": "Sales declined 15% in Q3",
    "objective": "Find root causes",
    "target": "Executive Board"
  }
}
```

**Response:** `{"data": {"task_id": "uuid", "status": "pending"}}` — poll via `/tasks/{task_id}`

Result includes: `analysis_markdown`, `goals`, `datasets_analyzed`, `model`

#### `POST /api/v1/strategic/suggest-questions`

Generate AI-suggested analysis questions based on data context.

```json
{
  "session_id": "uuid",
  "purpose": "Sales performance tracking",
  "role": "Analyst",
  "model": "qwen2.5:7b",
  "max_questions": 5
}
```

**Response:** `{"data": {"questions": ["What are the top products by revenue?", ...]}}`

#### `POST /api/v1/strategic/viz-recommend`

Get AI-recommended business-critical visualizations with pre-built Plotly charts.

```json
{
  "session_id": "uuid",
  "model": "qwen2.5:7b",
  "goals": {"problem": "...", "objective": "..."},
  "max_charts": 6
}
```

**Response:**
```json
{
  "data": {
    "recommendations": [
      {"chart_type": "bar", "dataset_name": "...", "x_column": "...", "y_column": "...", "reason": "..."}
    ],
    "plotly_charts": [
      {"title": "...", "plotly_json": "..."}
    ]
  }
}
```

---

### Database Source Endpoint (AI Engine)

#### `POST /api/v1/database/query`

Load data from an external database into an AI Engine session.

```json
{
  "session_id": null,
  "db_type": "postgresql",
  "db_host": "10.100.102.6",
  "db_port": 5432,
  "db_name": "analytics_db",
  "db_user": "reader",
  "db_password": "secret",
  "query": "SELECT * FROM sales WHERE year >= 2024",
  "dataset_name": "sales_2024"
}
```

**Response:**
```json
{
  "data": {
    "session_id": "uuid",
    "dataset_key": "db::sales_2024",
    "rows": 15420,
    "columns": 12,
    "column_names": ["id", "product", "amount", ...],
    "column_types": {"id": "int64", "product": "object", ...}
  }
}
```

---

### Pipeline Operations (AI Engine)

#### `POST /api/v1/pipeline/finalize`

Mark the current working DataFrame as the golden pipeline output.

```json
{
  "session_id": "uuid",
  "dataset_key": null
}
```

**Response:** `{"data": {"session_id": "uuid", "dataset_key": "pipeline_final", "rows": 5000, "columns": 15, "message": "Pipeline finalized..."}}`

#### `POST /api/v1/pipeline/lookup-column`

VLOOKUP — bring a column from another dataset by key matching.

```json
{
  "session_id": "uuid",
  "main_key": "orders.xlsx::Sheet1",
  "lookup_key": "customers.xlsx::Sheet1",
  "main_on": "customer_id",
  "lookup_on": "id",
  "value_column": "customer_name",
  "new_column_name": "customer"
}
```

**Response:** `{"data": {"dataset_key": "...", "new_column": "customer", "matched_rows": 4800, "total_rows": 5000, "preview": [...]}}`

#### `POST /api/v1/pipeline/ai-column`

Create a computed column from a natural-language description.

```json
{
  "session_id": "uuid",
  "description": "Calculate total revenue by multiplying Price and Quantity",
  "model": "qwen2.5:7b"
}
```

**Response:** `{"data": {"column_name": "total_revenue", "expression": "df['Price'] * df['Quantity']", "preview": [...]}}`

---

### Dashboard Export (AI Engine)

#### `POST /api/v1/dashboard/export-html`

Export dashboard as a self-contained HTML file with embedded Plotly.js charts.

```json
{
  "session_id": "uuid",
  "pages": [
    {
      "name": "Overview",
      "charts": [
        {"title": "Revenue by Region", "plotly_json": "..."}
      ]
    }
  ],
  "title": "Q3 Dashboard Report"
}
```

**Response:** Returns `text/html` content directly.

---

### Updated Page Configuration

**Available Pages (15 total):**
```
multi_file_loader, quick_excel_analysis, schema_explorer, data_quality,
data_cleaning, kpi_dashboard, visualization, custom_dashboard,
ai_analysis, executive_analysis, reports, chatbot,
rag_pipeline, contradiction_detection, decision_relevance
```

**Database-Hidden Pages:** When `workspace.data_source_type == "database"`:
- `data_cleaning` — cleaning pipeline is skipped for DB sources
- `data_upload` — no file upload needed
- `multi_file_loader` — data comes from DB query
- `quick_excel_analysis` — Excel-specific feature

**Color Palettes:** The `VisualizationRequest` now accepts `color_palette`:
- `vibrancy` (default): Bright, energetic colors
- `ocean`: Blue/teal gradients
- `sunset`: Orange/red warm tones
- `forest`: Green nature tones

---

### Endpoint Summary After Phase 4

| Service | Endpoint Count |
|---------|---------------|
| Service 1 (Backend) | 48 |
| Service 2 (AI Engine) | **55** (+11 new) |
| **Total** | **103** |

**New Endpoints (Service 2):**
| Method | Path | Feature |
|--------|------|---------|
| POST | `/strategic/goals` | Save business goals |
| GET | `/strategic/goals` | Get business goals |
| POST | `/strategic/analyze` | Full strategic analysis (background) |
| POST | `/strategic/suggest-questions` | AI question suggestions |
| POST | `/strategic/viz-recommend` | AI visualization recommendations |
| POST | `/database/query` | Load from external database |
| POST | `/pipeline/finalize` | Finalize golden dataset |
| POST | `/pipeline/lookup-column` | VLOOKUP column |
| POST | `/pipeline/ai-column` | AI-generated computed column |
| POST | `/dashboard/export-html` | Export dashboard as HTML |
| *(updated)* | `/visualization/generate` | Now supports `color_palette` |
