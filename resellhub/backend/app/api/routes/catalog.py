from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product, CustomerInteraction, User
from app.models.interaction import INTERACTION_WEIGHTS
from app.schemas import ProductOut, InteractionCreate
from app.core.deps import get_current_user
from app.core.cache import invalidate

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/products", response_model=list[ProductOut])
def list_products(
    category: str | None = None,
    q: str | None = Query(default=None, description="search title/brand"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Product)
    if category:
        query = query.filter(Product.category == category)
    if q:
        like = f"%{q}%"
        query = query.filter((Product.title.ilike(like)) | (Product.brand.ilike(like)))
    return query.offset(offset).limit(limit).all()


@router.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.post("/interactions", status_code=201)
def log_interaction(
    payload: InteractionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Customers call this as they browse/cart/purchase. Every call is a training
    signal for next retrain; cache is invalidated for this user so their next
    recommendation call doesn't serve a stale, pre-interaction result.
    """
    if not db.get(Product, payload.product_id):
        raise HTTPException(status_code=404, detail="Product not found")

    interaction = CustomerInteraction(
        user_id=user.id,
        product_id=payload.product_id,
        interaction_type=payload.interaction_type,
        weight=INTERACTION_WEIGHTS[payload.interaction_type],
    )
    db.add(interaction)
    db.commit()
    invalidate(f"reco:customer:{user.id}")
    return {"status": "logged"}
