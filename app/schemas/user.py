from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: str = "Pendamping"
    avatar_initials: str = "AP"
    elderly_name: Optional[str] = None
    created_at: Optional[str] = None

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    elderly_name: Optional[str] = Field(None, min_length=2, max_length=100)

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
