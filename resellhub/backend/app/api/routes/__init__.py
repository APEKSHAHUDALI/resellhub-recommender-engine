from fastapi import APIRouter

from app.api.routes import auth, catalog, recommendations, admin, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(catalog.router)
api_router.include_router(recommendations.router)
api_router.include_router(admin.router)
