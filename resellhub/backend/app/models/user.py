import enum
from datetime import datetime

from sqlalchemy import String, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class UserRole(str, enum.Enum):
    CUSTOMER = "customer"
    RESELLER = "reseller"
    ADMIN = "admin"


class User(Base):
    """
    A single users table serves both marketplace sides. `role` determines
    which recommendation surface (customer storefront vs reseller restocking)
    applies to this account.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.CUSTOMER, index=True)

    # Reseller-only business attributes, unused for customers
    store_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    preferred_categories: Mapped[str | None] = mapped_column(String(500), nullable=True)  # comma-separated

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    interactions = relationship("CustomerInteraction", back_populates="user", cascade="all, delete-orphan")
    stock_events = relationship("ResellerStockEvent", back_populates="reseller", cascade="all, delete-orphan")
