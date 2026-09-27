from harness.security import sign_payload, signed_headers, verify_signature


def test_sign_and_verify_roundtrip():
    body = b'{"job_id": "job_1"}'
    ts, sig = sign_payload("secret", body)
    ok, reason = verify_signature("secret", body, ts, sig)
    assert ok, reason


def test_tampered_body_is_rejected():
    body = b'{"job_id": "job_1"}'
    ts, sig = sign_payload("secret", body)
    ok, reason = verify_signature("secret", b'{"job_id": "job_2"}', ts, sig)
    assert not ok and reason == "signature mismatch"


def test_wrong_secret_is_rejected():
    body = b"x"
    ts, sig = sign_payload("secret", body)
    assert not verify_signature("other", body, ts, sig)[0]


def test_stale_timestamp_is_rejected():
    body = b"x"
    ts, sig = sign_payload("secret", body, timestamp="1000")
    ok, reason = verify_signature("secret", body, ts, sig, max_skew_seconds=60, now=5000)
    assert not ok and "window" in reason


def test_missing_headers_rejected():
    assert verify_signature("secret", b"x", None, None)[0] is False


def test_signed_headers_verify():
    body = b'{"a": 1}'
    headers = signed_headers("secret", body)
    ok, _ = verify_signature("secret", body, headers["X-Harness-Timestamp"], headers["X-Harness-Signature"])
    assert ok
