from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from app.core.security import decode_access_token
from app.core.supabase import db

security = HTTPBearer(auto_error=False)

async def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    if not credentials or not credentials.credentials:
        return None

    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        return None

    user_id = payload.get("id")
    email = payload.get("sub")
    if user_id is None and email is None:
        return None

    user = await db.get_user_by_id(user_id) if user_id else await db.get_user_by_email(email)
    if user and int(payload.get("ver", 0)) == int(user.get("token_version", 0)):
        return user
    return None

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Autentikasi gagal atau token tidak valid",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not credentials or not credentials.credentials:
        raise credentials_exception

    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("id")
    email = payload.get("sub")
    if user_id is None and email is None:
        raise credentials_exception

    user = await db.get_user_by_id(user_id) if user_id else await db.get_user_by_email(email)
    if user is None or int(payload.get("ver", 0)) != int(user.get("token_version", 0)):
        raise credentials_exception

    return user
