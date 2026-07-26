# ResellHub — Hybrid Recommendation Engine for a Two-Sided Resale Marketplace

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A production-structured recommendation system that powers **two different
recommendation surfaces from one shared engine**:

- **Customers** browsing a storefront → "recommended for you"
- **Resellers** deciding what to stock → "what to stock next" (margin-aware)

Built on **real transaction data** (not synthetic/fabricated), with a hybrid
collaborative-filtering + content-based + popularity-fallback model, an
honest offline evaluation harness, and the infrastructure around it
(auth, caching, background retraining, Docker, CI) that turns a notebook
model into something you could actually deploy.

**Built by:** B Sujan Kumar and Apeksha Sanjay Hudali

## Contents
- [Why this project](#why-this-is-a-good-placement-project-to-talk-about)
- [Evaluation results](#real-evaluation-results)
- [Architecture](#architecture)
- [Dataset & modeling decisions](#real-dataset--the-resellercustomer-split)
- [Project layout](#project-layout)
- [Running it](#running-it)
- [Interview talking points](#talking-points-for-interviews)
- [License](#license)

## Why this is a good placement project to talk about

- **Real data, not toy data.** Uses the UCI "Online Retail" dataset (541K
  real UK e-commerce transactions) instead of `Faker`-generated fake events.
- **A defensible modeling decision, not a shortcut.** The dataset has no
  "reseller" field, so resellers are *derived* from real bulk-purchasing
  behavior (median order-line quantity ≥ 12), a heuristic the dataset's own
  documentation supports ("many customers... are wholesalers"). That's a
  concrete story for an interview: *"the raw data didn't give me the label I
  needed, so I derived it from a documented, auditable behavioral signal."*
- **Cold-start is handled explicitly**, not ignored: the hybrid layer
  detects low-interaction users and reweights away from collaborative
  filtering toward content similarity and popularity, rather than serving
  garbage recommendations to new users.
- **It's evaluated, not just demoed.** `scripts/evaluate.py` runs a
  time-based train/test split and reports precision@k / recall@k / NDCG@k /
  hit-rate on real held-out data (numbers below).
- **It's built like a service**, with JWT auth, Redis-cached endpoints,
  scheduled retraining via Celery, health checks, Docker Compose, and CI.

## Real evaluation results

Run via `PYTHONPATH=. python scripts/evaluate.py` (time-based holdout,
train on the first 80% of interactions by date, test on the rest):

| Surface | Users evaluated | Precision@10 | Recall@10 | NDCG@10 | Hit rate@10 |
|---|---|---|---|---|---|
| Customer storefront | 858 | 0.187 | 0.076 | 0.211 | **66.3%** |
| Reseller restocking | 496 | 0.134 | 0.091 | 0.163 | **57.3%** |

*(Hit rate = % of users for whom at least one of the top-10 recommendations
matches something they actually went on to interact with after the cutoff
date — the most interview-friendly number to lead with.)*

## Architecture

```
┌─────────────┐      ┌──────────────────────────────────────────┐
│   React     │ HTTP │                FastAPI                    │
│  frontend   │─────▶│  /auth  /catalog  /recommendations  /admin│
└─────────────┘      └───────────┬──────────────────┬───────────┘
                                  │                  │
                          ┌───────▼───────┐   ┌──────▼──────┐
                          │   PostgreSQL   │   │    Redis     │
                          │ (users, catalog,│   │ (reco cache) │
                          │  interactions)  │   └──────┬──────┘
                          └───────┬────────┘          │
                                  │            ┌───────▼────────┐
                          ┌───────▼────────┐   │  Celery worker │
                          │ Recommender    │◀──│ + beat (nightly│
                          │ artifacts.pkl  │   │  retrain job)  │
                          │ (CF + content) │   └────────────────┘
                          └────────────────┘
```

**Why training is offline, not live**: the API only ever does matrix
lookups and cosine similarity against a pre-fitted model loaded from disk
(hot-reloaded on file-mtime change) — it never fits a model inside a
request. Training runs as a batch job (`app/recommender/train.py`),
schedulable nightly via Celery beat, or on-demand via `POST /admin/retrain`.

### The hybrid recommender, in one paragraph

Three signals are computed per user, min-max normalized (they live on
incomparable scales), and blended with configurable weights:
1. **Collaborative filtering** (`implicit` ALS) — "users/resellers like you
   also liked/stocked..." Two independent models: customer×product and
   reseller×product, each confidence-weighted (views < wishlist < cart <
   purchase for customers; `units_stocked × sell_through_rate` for
   resellers, so a reseller who overstocked a dud contributes less signal
   than one whose smaller batch fully sold through).
2. **Content-based** (TF-IDF over title/brand/category/condition/description
   + price) — makes cold-start possible, since a brand-new product has a
   content vector the moment it's catalogued, with zero interaction history
   required.
3. **Popularity fallback** with recency decay (half-life 14 days), used
   alone for true cold-start (zero-history) users.

The blend weight shifts automatically: warm users get all three signals;
users below `min_interactions_for_cf` (default 3) drop CF entirely and
redistribute its weight to content + popularity, since CF is unreliable with
too few signals.

## Real dataset & the reseller/customer split

Source: **"Online Retail"**, D. Chen, UCI Machine Learning Repository,
CC BY 4.0 (https://doi.org/10.24432/C5BW33) — real invoice-level transactions
from a UK-based online gift retailer, Dec 2010–Dec 2011.

`scripts/load_real_data.py` maps it onto the marketplace schema:

| Raw field | Used for |
|---|---|
| `Description` | Product title + input to TF-IDF content vectorizer |
| `UnitPrice` (median per SKU) | `Product.price` |
| `Quantity`, `InvoiceNo`, `InvoiceDate` | Interaction weight, purchase timestamps |
| `CustomerID` | Real user identity — segmented into customer/reseller |

Fields the raw data doesn't contain are *derived*, not invented, and every
derivation is commented in the loader:
- **Category** — a lightweight regex/keyword tagger over the real product
  titles (no category field exists in the source data).
- **Wholesale cost** — estimated at 55% of median unit price (a plausible
  gift-retail margin), used only for the reseller margin-boost business rule.
- **Sell-through rate** (resellers only) — approximated from how often a
  reseller re-orders the same SKU (repeat orders imply the earlier batch sold).
- **Login credentials** — the raw data has no emails/passwords, so demo
  accounts are synthesized (`{role}{CustomerID}@resellhub.demo`, password
  `demo12345` for every account) purely so the app is logged-in-demoable.
  The CustomerID and all behavioral data behind it are real.

## Project layout

```
resellhub/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, middleware, lifespan
│   │   ├── config.py                Centralized settings (pydantic-settings)
│   │   ├── database.py              SQLAlchemy engine/session
│   │   ├── models/                  User, Product, CustomerInteraction, ResellerStockEvent
│   │   ├── schemas.py                Pydantic request/response models
│   │   ├── api/routes/               auth, catalog, recommendations, admin, health
│   │   ├── core/                     security (JWT/bcrypt), cache (Redis), deps (auth guards)
│   │   ├── services/
│   │   │   ├── recommendation_service.py   ties models + cache + business rules together
│   │   │   └── tasks.py                     Celery app + nightly retrain schedule
│   │   └── recommender/
│   │       ├── collaborative.py      implicit ALS wrapper (used for both surfaces)
│   │       ├── content_based.py      TF-IDF + cosine similarity
│   │       ├── popularity.py         recency-decayed popularity fallback
│   │       ├── hybrid.py             blending + cold-start weight switching
│   │       ├── evaluation.py         precision@k / recall@k / NDCG@k
│   │       └── train.py              offline training orchestrator
│   ├── scripts/
│   │   ├── download_dataset.sh       fetch the real CSV
│   │   ├── load_real_data.py         ETL: real CSV -> marketplace schema
│   │   └── evaluate.py               offline evaluation on held-out real data
│   ├── tests/                        pytest: recommender unit tests + API tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                         React + Vite: login, customer/reseller dashboards, catalog
├── docker-compose.yml                postgres + redis + backend + celery + frontend
└── .github/workflows/ci.yml          backend tests, frontend build, docker build
```

## Running it

### Option A — Docker Compose (production-shaped)
```bash
cd resellhub
./backend/scripts/download_dataset.sh    # fetches the real dataset into backend/data/
docker compose up -d postgres redis
docker compose run --rm data-loader      # loads real data + trains models (~20s)
docker compose up -d backend celery-worker celery-beat frontend
# API:      http://localhost:8000/api/docs
# Frontend: http://localhost:3000
```

### Option B — local dev (fastest to iterate)
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./scripts/download_dataset.sh
PYTHONPATH=. python scripts/load_real_data.py     # ~15-20s for the full 541K rows
PYTHONPATH=. python -m app.recommender.train
uvicorn app.main:app --reload

# separately
cd frontend
npm install && npm run dev   # http://localhost:5173, proxies /api to :8000
```

Demo login: any email like `customer12346@resellhub.demo` or
`reseller12347@resellhub.demo`, password `demo12345` (see loader output /
query the `users` table for other real IDs).

### Tests
```bash
cd backend
PYTHONPATH=. pytest tests/ -v --cov=app
```

## Talking points for interviews

- **"Why hybrid instead of just collaborative filtering?"** Pure CF fails
  completely on cold-start users/items — no history, no recommendation. The
  content-based layer and popularity fallback exist specifically to cover
  that gap, and the weight-switching logic in `hybrid.py` is the mechanism
  that makes the handoff automatic rather than a special-cased branch.
- **"How do you know it's actually good?"** Point to `scripts/evaluate.py`
  and the time-based holdout numbers above — not vibes, a real protocol.
- **"How would this scale?"** Training is already decoupled from serving
  (offline batch job → pickled artifacts → hot-reloaded by mtime), so the
  API's request path is O(1) lookups + cosine similarity, never a live
  model fit. Redis caches full responses per user. The obvious next steps
  (approximate nearest neighbors for the content vectors, a real feature
  store, A/B testing infrastructure) are natural "if I had more time" answers.
- **"What was the hardest bug?"** bcrypt hashing is deliberately slow
  (~0.3s/call) — the initial data loader hashed the demo password once per
  user (4,362 times), which alone accounted for ~20 minutes of runtime. The
  fix was hashing once and reusing the hash across all demo accounts, since
  they intentionally share one password. Good illustration of profiling
  before optimizing: pandas and the DB writes were never the bottleneck.

## Acknowledgments

Dataset: **"Online Retail"**, D. Chen, UCI Machine Learning Repository,
CC BY 4.0 (https://doi.org/10.24432/C5BW33). Used here for a non-commercial,
educational/portfolio project.

Authors : B Sujan Kumar and Apeksha Sanjay Hudali. 

## License

MIT — see [LICENSE](./LICENSE). Copyright (c) 2026 B Sujan Kumar and
Apeksha Sanjay Hudali. 
