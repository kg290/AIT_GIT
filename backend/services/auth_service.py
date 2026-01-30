"""
Authentication Service
JWT-based authentication for hospital staff
"""
import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import User, UserRole, AuditLog


# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours

# Security
security = HTTPBearer(auto_error=False)


class AuthService:
    """Authentication and authorization service"""
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt directly"""
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        try:
            password_bytes = plain_password.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate a user by username and password"""
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    def get_current_user(
        self, 
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
    ) -> User:
        """Get current user from JWT token"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        if not credentials:
            raise credentials_exception
        
        token = credentials.credentials
        payload = self.decode_token(token)
        
        if payload is None:
            raise credentials_exception
        
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        user = db.query(User).filter(User.username == username).first()
        if user is None or not user.is_active:
            raise credentials_exception
        
        # Attach user info to request for audit logging
        request.state.user = user
        
        return user
    
    def require_roles(self, *allowed_roles: UserRole):
        """Dependency to require specific roles"""
        def role_checker(current_user: User = Depends(self.get_current_user)):
            if current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
                )
            return current_user
        return role_checker


# Global auth service instance
auth_service = AuthService()


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """FastAPI dependency for getting current user"""
    return auth_service.get_current_user(request, credentials, db)


def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """FastAPI dependency for optionally getting current user"""
    if not credentials:
        return None
    try:
        return auth_service.get_current_user(request, credentials, db)
    except HTTPException:
        return None


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_doctor_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require doctor or admin role"""
    if current_user.role not in [UserRole.ADMIN, UserRole.DOCTOR]:
        raise HTTPException(status_code=403, detail="Doctor or admin access required")
    return current_user


def require_clinical_staff(current_user: User = Depends(get_current_user)) -> User:
    """Require clinical staff (doctor, pharmacist, nurse)"""
    allowed = [UserRole.ADMIN, UserRole.DOCTOR, UserRole.PHARMACIST, UserRole.NURSE]
    if current_user.role not in allowed:
        raise HTTPException(status_code=403, detail="Clinical staff access required")
    return current_user


class AuditService:
    """HIPAA-compliant audit logging service"""
    
    @staticmethod
    def log(
        db: Session,
        action: str,
        resource_type: str,
        resource_id: str = None,
        description: str = None,
        old_values: dict = None,
        new_values: dict = None,
        user: User = None,
        request: Request = None,
        success: bool = True,
        error_message: str = None
    ):
        """Create an audit log entry"""
        log_entry = AuditLog(
            user_id=user.id if user else None,
            username=user.username if user else "system",
            user_role=user.role.value if user else "system",
            ip_address=request.client.host if request else None,
            user_agent=request.headers.get("user-agent", "")[:500] if request else None,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            description=description,
            old_values=old_values,
            new_values=new_values,
            success=success,
            error_message=error_message
        )
        db.add(log_entry)
        db.commit()
        return log_entry


audit_service = AuditService()
