import os
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret key for JWT signing. In production (MHARS_REQUIRE_AUTH=true) a strong
# secret MUST be supplied via MHARS_JWT_SECRET — the default dev key is rejected
# so tokens cannot be forged by anyone who has read the source.
_DEFAULT_DEV_SECRET = "***REDACTED_DEV_SECRET***"
SECRET_KEY = os.environ.get("MHARS_JWT_SECRET", _DEFAULT_DEV_SECRET)

def _auth_required() -> bool:
    return os.environ.get("MHARS_REQUIRE_AUTH", "false").strip().lower() in ("1", "true", "yes", "on")

if _auth_required() and SECRET_KEY == _DEFAULT_DEV_SECRET:
    raise RuntimeError(
        "MHARS_REQUIRE_AUTH is enabled but MHARS_JWT_SECRET is unset. "
        "Set a strong random MHARS_JWT_SECRET before running in production."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def _load_users() -> Dict[str, Any]:
    if not os.path.exists(USERS_FILE):
        _seed_default_users()
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def _save_users(users: Dict[str, Any]):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def _seed_default_users():
    """Create default users with varying roles if the users file doesn't exist."""
    users = {
        "admin": {
            "password_hash": pwd_context.hash("admin123"),
            "role": "admin",
            "created_at": time.time()
        },
        "operator": {
            "password_hash": pwd_context.hash("oper123"),
            "role": "operator",
            "created_at": time.time()
        },
        "viewer": {
            "password_hash": pwd_context.hash("view123"),
            "role": "viewer",
            "created_at": time.time()
        }
    }
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"username": username, "role": user["role"]}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_user(username: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    user = users.get(username)
    if user:
        return {"username": username, "role": user["role"]}
    return None

def list_users() -> List[Dict[str, Any]]:
    users = _load_users()
    return [{"username": k, "role": v["role"], "created_at": v.get("created_at")} for k, v in users.items()]

def update_user_role(username: str, new_role: str) -> bool:
    users = _load_users()
    if username in users:
        users[username]["role"] = new_role
        _save_users(users)
        return True
    return False

def delete_user(username: str) -> bool:
    users = _load_users()
    if username in users:
        del users[username]
        _save_users(users)
        return True
    return False

def create_user(username: str, password: str, role: str) -> bool:
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": get_password_hash(password),
        "role": role,
        "created_at": time.time()
    }
    _save_users(users)
    return True
