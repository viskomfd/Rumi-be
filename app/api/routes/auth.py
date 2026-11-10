from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.user import PasswordChange, UserRegister, UserLogin, UserResponse, UserUpdate, Token
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.supabase import db
from app.api.deps import get_current_user
from typing import Dict, Any

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserRegister):
    existing_user = await db.get_user_by_email(user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email sudah terdaftar. Silakan gunakan email lain atau login."
        )

    # Initial avatar
    initials = "".join([part[0].upper() for part in user_in.name.split()[:2]]) or "AP"

    user_data = {
        "name": user_in.name,
        "email": user_in.email,
        "hashed_password": get_password_hash(user_in.password),
        "role": "Pendamping",
        "avatar_initials": initials
    }

    created_user = await db.create_user(user_data)
    token = create_access_token({
        "sub": created_user["email"],
        "id": created_user["id"],
        "ver": created_user.get("token_version", 0),
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**created_user)
    }

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    user = await db.get_user_by_email(credentials.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau kata sandi salah."
        )

    if not verify_password(credentials.password, user.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau kata sandi salah."
        )

    token = create_access_token({
        "sub": user["email"],
        "id": user["id"],
        "ver": user.get("token_version", 0),
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user)
    }

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return UserResponse(**current_user)

@router.patch("/me", response_model=UserResponse)
async def update_me(
    user_in: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    changes = user_in.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tidak ada perubahan.")

    if "name" in changes:
        changes["name"] = changes["name"].strip()
        if len(changes["name"]) < 2:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Nama minimal 2 karakter.")
        changes["avatar_initials"] = "".join(
            part[0].upper() for part in changes["name"].split()[:2]
        ) or "AP"
    if "email" in changes:
        changes["email"] = str(changes["email"]).lower().strip()
        existing_user = await db.get_user_by_email(changes["email"])
        if existing_user and existing_user["id"] != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email sudah digunakan.")
    if "elderly_name" in changes:
        changes["elderly_name"] = changes["elderly_name"].strip()
        if len(changes["elderly_name"]) < 2:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Nama lansia minimal 2 karakter.")

    updated_user = await db.update_user(current_user["id"], changes)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Profil gagal diperbarui.")
    return UserResponse(**updated_user)

@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    password_in: PasswordChange,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if not verify_password(password_in.current_password, current_user.get("hashed_password", "")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password saat ini salah.")
    updated_user = await db.update_user(current_user["id"], {
        "hashed_password": get_password_hash(password_in.new_password),
        "token_version": int(current_user.get("token_version", 0)) + 1,
    })
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password gagal diperbarui.")

@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
async def logout_all(current_user: Dict[str, Any] = Depends(get_current_user)):
    updated_user = await db.update_user(current_user["id"], {
        "token_version": int(current_user.get("token_version", 0)) + 1,
    })
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sesi gagal dihentikan.")
