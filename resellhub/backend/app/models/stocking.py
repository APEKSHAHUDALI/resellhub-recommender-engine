from datetime import datetime

from sqlalchemy import ForeignKey, DateTime, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ResellerStockEvent(Base):
    """
    A reseller stocking a product, and how well it sold. `sell_through_rate`
    (units_sold / units_stocked) is the implicit-feedback "confidence" signal
    for the reseller-side collaborative filter: it tells us not just *that*
    a reseller stocked something, but whether it was a good call.
    """
    __tablename__ = "reseller_stock_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    reseller_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    units_stocked: Mapped[int] = mapped_column(Integer, default=0)
    units_sold: Mapped[int] = mapped_column(Integer, default=0)
    sell_through_rate: Mapped[float] = mapped_column(Float, default=0.0)
    margin: Mapped[float] = mapped_column(Float, default=0.0)  # (price - wholesale_cost) / price
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    reseller = relationship("User", back_populates="stock_events")
