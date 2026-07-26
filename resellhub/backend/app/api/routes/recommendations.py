from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.deps import get_current_user, require_role
from app.models import User, UserRole
from app.schemas import RecommendationResponse, RecommendationItem
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/customer/me", response_model=RecommendationResponse)
def my_customer_recommendations(
    n: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(require_role(UserRole.CUSTOMER)),
):
    service = RecommendationService(db)
    items = service.recommend_for_customer(user.id, n=n)
    return RecommendationResponse(
        user_id=user.id, surface="customer", items=[RecommendationItem(**i) for i in items]
    )


@router.get("/reseller/me", response_model=RecommendationResponse)
def my_reseller_recommendations(
    n: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(require_role(UserRole.RESELLER)),
):
    service = RecommendationService(db)
    items = service.recommend_for_reseller(user.id, n=n)
    return RecommendationResponse(
        user_id=user.id, surface="reseller_stocking", items=[RecommendationItem(**i) for i in items]
    )


@router.get("/products/{product_id}/similar", response_model=list[RecommendationItem])
def similar_products(product_id: int, n: int = Query(default=8, ge=1, le=30), db: Session = Depends(get_db)):
    service = RecommendationService(db)
    items = service.similar_products(product_id, n=n)
    return [RecommendationItem(**i) for i in items]
