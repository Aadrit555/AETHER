from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from aether.core.config import settings
from aether.core.database import connect


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        _, salt_hex, digest_hex = encoded.split("$", 2)
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def create_user(email: str, password: str) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO users(email, password_hash) VALUES(?, ?)", (email.lower().strip(), hash_password(password)))
        return int(cur.lastrowid)


def authenticate(email: str, password: str) -> int | None:
    with connect() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return int(row["id"])
    return None


def token_for(user_id: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_exp_minutes)
    return jwt.encode({"sub": str(user_id), "exp": exp}, settings.jwt_secret, algorithm="HS256")


def user_from_token(token: str) -> int:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    return int(payload["sub"])
