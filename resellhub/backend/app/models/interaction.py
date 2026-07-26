import enum
from datetime import datetime

from sqlalchemy import ForeignKey, DateTime, Enum, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class InteractionType(str, enum.Enum):
    VIEW = "view"
    CART = "cart"
    PURCHASE = "purchase"
    WISHLIST = "wishlist"


# Implicit-feedback confidence weights per interaction type.
# Used to build the weighted user-item matrix for collaborative filtering.
INTERACTION_WEIGHTS = {
    InteractionType.VIEW: 1.0,
    InteractionType.WISHLIST: 2.0,
    InteractionType.CART: 3.0,
    InteractionType.PURCHASE: 5.0,
}


class CustomerInteraction(Base):
    """Implicit feedback event: a customer viewed/carted/purchased a product."""
    __tablename__ = "customer_interactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    interaction_type: Mapped[InteractionType] = mapped_column(Enum(InteractionType))
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="interactions")
