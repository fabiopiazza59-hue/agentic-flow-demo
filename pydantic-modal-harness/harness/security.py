"""
HMAC signing and verification for webhook payloads.

Both directions use the same scheme:
- Modal worker  -> harness  (`POST /webhooks/modal`)
- harness       -> client   (optional `callback_url` on a run)

Signature = HMAC-SHA256(secret, f"{timestamp}.{raw_body}") as hex, sent as
`X-Harness-Signature: sha256=<hex>` with `X-Harness-Timestamp: <unix seconds>`.
The timestamp is bound into the signature to limit replay windows.
"""

from __future__ import annotations

import hashlib
import hmac
import time

SIGNATURE_HEADER = "X-Harness-Signature"
TIMESTAMP_HEADER = "X-Harness-Timestamp"


def sign_payload(secret: str, body: bytes, timestamp: str | None = None) -> tuple[str, str]:
    """Return (timestamp, signature_header_value) for a raw request body."""
    ts = timestamp or str(int(time.time()))
    mac = hmac.new(secret.encode("utf-8"), f"{ts}.".encode("utf-8") + body, hashlib.sha256)
    return ts, f"sha256={mac.hexdigest()}"


def signed_headers(secret: str, body: bytes) -> dict[str, str]:
    ts, sig = sign_payload(secret, body)
    return {TIMESTAMP_HEADER: ts, SIGNATURE_HEADER: sig, "Content-Type": "application/json"}


def verify_signature(
    secret: str,
    body: bytes,
    timestamp: str | None,
    signature: str | None,
    max_skew_seconds: int = 300,
    now: float | None = None,
) -> tuple[bool, str]:
    """Constant-time verification. Returns (ok, reason)."""
    if not timestamp or not signature:
        return False, "missing signature headers"
    try:
        ts_int = int(timestamp)
    except ValueError:
        return False, "invalid timestamp"
    current = now if now is not None else time.time()
    if abs(current - ts_int) > max_skew_seconds:
        return False, "timestamp outside allowed window"
    _, expected = sign_payload(secret, body, timestamp)
    if not hmac.compare_digest(expected, signature):
        return False, "signature mismatch"
    return True, "ok"
