"""
Pytest configuration and fixtures.
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.main import app
from backend.db.database import create_tables, SessionLocal


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create tables before any tests run."""
    create_tables()
    yield


@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient with session-scoped setup.
    Model loads once for all tests.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def db():
    session = SessionLocal()
    yield session
    session.close()
