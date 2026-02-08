"""
Authentication and tenant isolation for MCP Gateway Gateway.
"""

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import hashlib
import time

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Tenant(BaseModel):
    """Tenant information extracted from API key."""
    tenant_id: str
    name: str
    tier: str  # "standard", "enterprise"
    rate_limit: int  # calls per minute


# In-memory API key store (would be database in production)
API_KEYS: dict[str, Tenant] = {
    "ok-test-key-001": Tenant(
        tenant_id="tenant-001",
        name="Pharma Corp",
        tier="enterprise",
        rate_limit=1000
    ),
    "ok-test-key-002": Tenant(
        tenant_id="tenant-002",
        name="Biotech Startup",
        tier="standard",
        rate_limit=100
    ),
    "ok-demo-key": Tenant(
        tenant_id="demo",
        name="Demo Tenant",
        tier="enterprise",
        rate_limit=10000
    ),
}

# Rate limiting state
_rate_limit_state: dict[str, list[float]] = {}


def _check_rate_limit(tenant_id: str, limit: int) -> bool:
    """Check if tenant is within rate limit."""
    now = time.time()
    window = 60  # 1 minute window

    if tenant_id not in _rate_limit_state:
        _rate_limit_state[tenant_id] = []

    # Clean old entries
    _rate_limit_state[tenant_id] = [
        t for t in _rate_limit_state[tenant_id]
        if now - t < window
    ]

    if len(_rate_limit_state[tenant_id]) >= limit:
        return False

    _rate_limit_state[tenant_id].append(now)
    return True


async def get_current_tenant(api_key: Optional[str] = Security(api_key_header)) -> Tenant:
    """Validate API key and return tenant info."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )

    tenant = API_KEYS.get(api_key)
    if not tenant:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    # Check rate limit
    if not _check_rate_limit(tenant.tenant_id, tenant.rate_limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {tenant.rate_limit} calls/minute."
        )

    return tenant


async def get_optional_tenant(api_key: Optional[str] = Security(api_key_header)) -> Optional[Tenant]:
    """Get tenant if API key provided, otherwise None (for public endpoints)."""
    if not api_key:
        return None
    return API_KEYS.get(api_key)
