from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Product(Base):
    """
    Catalog item. Resellers stock these; customers buy them.
    Text fields (title/description/category/brand/condition) feed the
    content-based recommender's TF-IDF vectorizer.
    """
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(100), index=True)
    subcategory: Mapped[str] = mapped_column(String(100), default="")
    brand: Mapped[str] = mapped_column(String(100), default="")
    condition: Mapped[str] = mapped_column(String(50), default="used_good")  # new, used_like_new, used_good, used_fair
    price: Mapped[float] = mapped_column(Float)
    wholesale_cost: Mapped[float] = mapped_column(Float, default=0.0)
    inventory_count: Mapped[int] = mapped_column(Integer, default=0)
    popularity_score: Mapped[float] = mapped_column(Float, default=0.0)  # precomputed, decays over time
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def text_blob(self) -> str:
        """Concatenated text used as input to the content-based vectorizer."""
        return " ".join(filter(None, [self.title, self.brand, self.category, self.subcategory, self.condition, self.description]))
