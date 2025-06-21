# System Architecture Overview

```
            +-------------+                           +-------------------+
            | Next.js UI  |  REST/Fetch (HTTPS)       |  API Routes       |
            |  (React)    | ─────────────────────────►|  (/app/api/...)   |
            +------+------+                           +---------+---------+
                   ▲                                            |
                   │ WebSocket (dev preview)                    │
                   │                                            │ calls
                   │                                            ▼
     +-------------+-----------+                   +--------------------------+
     | Geospatial Visualization|  Esri JS API      | SHAP Micro-service       |
     | (MapView, LayerManager) |◄──────────────────┤ (Python / Flask)         |
     +-------------------------+                   +--------------------------+
                                                        ▲          ▲
                         feature vectors, SHAP JSON     │          │ model files
                                                        │          │
                  +------------------+                  │          │
                  | Redis / Postgres |◄─────────────────┘          |
                  +------------------+                             |
                                                                   |
                  +------------------+                             |
                  |  Object Storage  |◄────────────────────────────┘
                  +------------------+
```

## Key Packages

| Path | Responsibility |
|------|----------------|
| `components/` | React components incl. map, chat, pop-ups |
| `app/api/claude/` | Edge/Node API routes + AI persona prompts |
| `lib/` | Client-side helpers (query classifier, visualization factory) |
| `scripts/` | dev / ops scripts (smoke tests, data loaders) |
| `shap-microservice/` | Python service performing XGBoost inference + SHAP |

## Data Flow (persona example)
1. UI sends query + selected `persona` to `/api/claude/generate-response`.  
2. Route loads persona prompt, builds system prompt, streams to Anthropic.  
3. AI response returned to UI, identifiers parsed for clickable map features.  
4. User may trigger visualization; UI collects features, calls SHAP service if needed.  
5. Resulting layer rendered via Esri JS API.

## Deployment Units
* **Render** – hosts Next.js app + Node functions.  
* **Render (micro-service)** – containerised Python SHAP service.  
* **Redis Cloud** – caching layer for expensive computations.  
* **AWS S3** (or Vercel Blob) – temporary feature blobs uploaded from browser. 