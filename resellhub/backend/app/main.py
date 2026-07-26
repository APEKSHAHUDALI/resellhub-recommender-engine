import logging
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import Base, engine
from app.api.routes import api_router

settings = get_settings()

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("resellhub")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # In production, schema changes go through Alembic migrations - this call
    # is a convenience for local/dev/demo so the app is runnable immediately.
    if settings.environment != "production":
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Hybrid recommendation engine for a two-sided resale marketplace "
                 "(customer storefront recommendations + reseller restocking recommendations).",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response


app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running", "docs": "/api/docs"}
