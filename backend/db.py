"""SQLAlchemy engine/session setup for the SQLite-backed persistence layer."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _ensure_sqlite_directory(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    db_path = Path(database_url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_directory(settings.database_url)

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Creates all tables. Called from models/__init__.py, once every ORM
    model in that package has finished defining its class (and so has
    registered itself on Base.metadata) — never from here, since db.py is
    itself imported from inside models/orm.py (`from ..db import Base`) and
    calling back into `models` from here would re-enter that partially
    -initialized module and silently skip whichever classes haven't been
    defined yet, creating tables non-deterministically depending on import
    order."""
    Base.metadata.create_all(engine)
