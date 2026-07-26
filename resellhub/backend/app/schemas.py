from pydantic import BaseModel, EmailStr, Field

from app.models.user import UserRole
from app.models.interaction import InteractionType


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str
    role: UserRole = UserRole.CUSTOMER
    store_name: str | None = None


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: UserRole
    store_name: str | None = None

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ProductOut(BaseModel):
    id: int
    sku: str
    title: str
    category: str
    brand: str
    condition: str
    price: float
    inventory_count: int

    model_config = {"from_attributes": True}


class InteractionCreate(BaseModel):
    product_id: int
    interaction_type: InteractionType


class RecommendationItem(BaseModel):
    product_id: int
    sku: str
    title: str
    category: str
    price: float
    condition: str
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    user_id: int
    surface: str
    items: list[RecommendationItem]
    cached: bool = False


class TrainingSummary(BaseModel):
    products: int
    customer_interactions: int
    reseller_events: int
    customer_cf_coverage: dict
    reseller_cf_coverage: dict
