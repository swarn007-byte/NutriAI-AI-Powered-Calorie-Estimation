"""Minimal JWT auth (design.md §10 — "JWT-based, minimal").

No OAuth providers, no refresh-token rotation: those are §16 scale-up items.
Guest sessions exist so the first analysis needs no signup (design.md §13.1).

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library rather than
adding a bcrypt/argon2 dependency at this stage.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from config import settings
from db import User, get_session, get_user, new_id

_PBKDF2_ROUNDS = 240_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        algo, rounds, salt_hex, digest_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(expected, candidate)


def create_token(user_id: str, *, is_guest: bool = False) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "guest": is_guest,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.jwt_ttl_seconds)).timestamp()),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired — please sign in again.")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication token.")


def create_guest_user(session: Session) -> User:
    user = User(
        id=new_id(),
        email=None,
        name="Guest",
        is_guest=True,
        preferences={"calorie_goal": 2000, "plate_diameter_cm": settings.default_plate_diameter_cm},
    )
    session.add(user)
    session.flush()
    return user


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def current_user(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> User:
    """Required auth. Any caller without a valid token gets a 401."""
    token = _bearer(authorization)
    if not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    user = get_user(session, str(payload.get("sub", "")))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "This session no longer exists.")
    return user


def current_user_or_guest(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> User:
    """Auth that never blocks the first try — mints a guest if there's no token.

    design.md §13.1: "No login wall before the first try, if possible."
    """
    token = _bearer(authorization)
    if token:
        payload = decode_token(token)
        user = get_user(session, str(payload.get("sub", "")))
        if user is not None:
            return user
    return create_guest_user(session)
