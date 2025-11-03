"""
Authentication and Authorization Service for Project Argus.
Provides JWT-based authentication, MFA, and role-based access control.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
import pyotp
import qrcode
import io
import base64
from passlib.context import CryptContext
import redis
import asyncpg
import asyncio
from contextlib import asynccontextmanager

# Import shared models and interfaces
import sys
sys.path.append('/app/shared')
from models.user import User, UserRole, Permission, ROLE_PERMISSIONS
from interfaces.security import IAccessController

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
MFA_ISSUER = "Project Argus"

# Database and Redis configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://argus:argus@db:5432/argus")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# Global connections
db_pool = None
redis_client = None


class LoginRequest(BaseModel):
    username: str
    password: str
    mfa_token: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]


class MFASetupResponse(BaseModel):
    secret: str
    qr_code: str
    backup_codes: List[str]


class UserCreateRequest(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    role: str
    phone_number: Optional[str] = None
    department: Optional[str] = None
    badge_number: Optional[str] = None


class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    active: Optional[bool] = None
    phone_number: Optional[str] = None
    department: Optional[str] = None
    badge_number: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class AuthenticationService(IAccessController):
    """Implementation of authentication and authorization service."""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client
    
    async def authenticate_user(self, username: str, password: str, 
                               mfa_token: Optional[str] = None) -> Optional[User]:
        """Authenticate user with username/password and optional MFA."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get user from database
                row = await conn.fetchrow(
                    "SELECT * FROM users WHERE username = $1 AND active = true",
                    username
                )
                
                if not row:
                    logger.warning(f"Authentication failed: user {username} not found")
                    return None
                
                user = User(
                    id=row['id'],
                    username=row['username'],
                    email=row['email'],
                    full_name=row['full_name'],
                    password_hash=row['password_hash'],
                    salt=row['salt'],
                    mfa_enabled=row['mfa_enabled'],
                    mfa_secret=row['mfa_secret'],
                    role=UserRole(row['role']),
                    active=row['active'],
                    locked=row['locked'],
                    failed_login_attempts=row['failed_login_attempts'],
                    locked_until=row['locked_until'],
                    last_login=row['last_login'],
                    created_at=row['created_at']
                )
                
                # Check if account is locked
                if user.is_account_locked():
                    logger.warning(f"Authentication failed: account {username} is locked")
                    return None
                
                # Verify password
                if not pwd_context.verify(password, user.password_hash):
                    # Record failed login
                    user.record_login(False)
                    await self._update_user_login_info(user)
                    logger.warning(f"Authentication failed: invalid password for {username}")
                    return None
                
                # Check MFA if enabled
                if user.mfa_enabled:
                    if not mfa_token:
                        logger.warning(f"Authentication failed: MFA token required for {username}")
                        return None
                    
                    if not self._verify_mfa_token(user.mfa_secret, mfa_token):
                        user.record_login(False)
                        await self._update_user_login_info(user)
                        logger.warning(f"Authentication failed: invalid MFA token for {username}")
                        return None
                
                # Successful authentication
                user.record_login(True)
                await self._update_user_login_info(user)
                
                # Log successful authentication
                await self.audit_access(
                    user.id, "authenticate", "system", True,
                    {"method": "password_mfa" if user.mfa_enabled else "password"}
                )
                
                logger.info(f"User {username} authenticated successfully")
                return user
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def authorize_action(self, user: User, permission: Permission, 
                              resource_id: Optional[str] = None) -> bool:
        """Check if user is authorized to perform action."""
        try:
            # Check if user has the required permission
            if not user.has_permission(permission):
                await self.audit_access(
                    user.id, f"authorize_{permission.value}", 
                    resource_id or "system", False,
                    {"reason": "insufficient_permissions"}
                )
                return False
            
            # Additional resource-specific checks can be added here
            # For example, camera access restrictions, incident ownership, etc.
            
            await self.audit_access(
                user.id, f"authorize_{permission.value}", 
                resource_id or "system", True
            )
            return True
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
    
    async def create_session(self, user: User, client_info: Dict[str, Any]) -> str:
        """Create JWT session token for user."""
        try:
            # Create JWT payload
            payload = {
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in user.get_permissions()],
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
            }
            
            # Generate JWT token
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            
            # Store session in Redis with expiration
            session_data = {
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "created_at": datetime.utcnow().isoformat(),
                "client_info": client_info
            }
            
            await self.redis.setex(
                f"session:{token}",
                timedelta(hours=JWT_EXPIRATION_HOURS),
                str(session_data)
            )
            
            # Log session creation
            await self.audit_access(
                user.id, "create_session", "system", True,
                {"client_info": client_info}
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
    
    async def validate_session(self, session_token: str) -> Optional[User]:
        """Validate JWT session token and return user."""
        try:
            # Decode JWT token
            payload = jwt.decode(session_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check if session exists in Redis
            session_exists = await self.redis.exists(f"session:{session_token}")
            if not session_exists:
                return None
            
            # Get user from database
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM users WHERE id = $1 AND active = true",
                    payload["user_id"]
                )
                
                if not row:
                    return None
                
                user = User(
                    id=row['id'],
                    username=row['username'],
                    email=row['email'],
                    full_name=row['full_name'],
                    role=UserRole(row['role']),
                    active=row['active'],
                    locked=row['locked'],
                    mfa_enabled=row['mfa_enabled'],
                    created_at=row['created_at'],
                    last_login=row['last_login']
                )
                
                return user
                
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None
    
    async def revoke_session(self, session_token: str) -> bool:
        """Revoke user session."""
        try:
            # Remove session from Redis
            result = await self.redis.delete(f"session:{session_token}")
            
            # Try to get user info for audit log
            try:
                payload = jwt.decode(session_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                await self.audit_access(
                    payload["user_id"], "revoke_session", "system", True
                )
            except:
                pass  # Token might be invalid, but we still want to remove it
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Session revocation error: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for user."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT role, custom_permissions FROM users WHERE id = $1",
                    user_id
                )
                
                if not row:
                    return []
                
                role = UserRole(row['role'])
                role_permissions = ROLE_PERMISSIONS.get(role, set())
                
                # Add custom permissions if any
                custom_perms = set()
                if row['custom_permissions']:
                    custom_perms = {Permission(p) for p in row['custom_permissions']}
                
                return list(role_permissions.union(custom_perms))
                
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return []
    
    async def assign_role(self, user_id: str, role: UserRole) -> bool:
        """Assign role to user."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE users SET role = $1 WHERE id = $2",
                    role.value, user_id
                )
                
                await self.audit_access(
                    user_id, "assign_role", "system", True,
                    {"new_role": role.value}
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Error assigning role: {e}")
            return False
    
    async def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant specific permission to user."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get current custom permissions
                row = await conn.fetchrow(
                    "SELECT custom_permissions FROM users WHERE id = $1",
                    user_id
                )
                
                if not row:
                    return False
                
                custom_perms = set(row['custom_permissions'] or [])
                custom_perms.add(permission.value)
                
                await conn.execute(
                    "UPDATE users SET custom_permissions = $1 WHERE id = $2",
                    list(custom_perms), user_id
                )
                
                await self.audit_access(
                    user_id, "grant_permission", "system", True,
                    {"permission": permission.value}
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Error granting permission: {e}")
            return False
    
    async def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke specific permission from user."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get current custom permissions
                row = await conn.fetchrow(
                    "SELECT custom_permissions FROM users WHERE id = $1",
                    user_id
                )
                
                if not row:
                    return False
                
                custom_perms = set(row['custom_permissions'] or [])
                custom_perms.discard(permission.value)
                
                await conn.execute(
                    "UPDATE users SET custom_permissions = $1 WHERE id = $2",
                    list(custom_perms), user_id
                )
                
                await self.audit_access(
                    user_id, "revoke_permission", "system", True,
                    {"permission": permission.value}
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Error revoking permission: {e}")
            return False
    
    async def audit_access(self, user_id: str, action: str, resource: str, 
                          success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Log access attempt for audit."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_logs (user_id, action, resource, success, details, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, action, resource, success, details or {}, datetime.utcnow())
                
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
    
    async def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active sessions, optionally filtered by user."""
        try:
            # Get all session keys from Redis
            session_keys = await self.redis.keys("session:*")
            sessions = []
            
            for key in session_keys:
                session_data = await self.redis.get(key)
                if session_data:
                    try:
                        session_info = eval(session_data)  # Note: In production, use proper JSON parsing
                        if not user_id or session_info.get("user_id") == user_id:
                            sessions.append({
                                "token": key.decode().replace("session:", ""),
                                **session_info
                            })
                    except:
                        continue
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    async def enforce_password_policy(self, password: str) -> tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def _verify_mfa_token(self, secret: str, token: str) -> bool:
        """Verify TOTP MFA token."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except:
            return False
    
    async def _update_user_login_info(self, user: User) -> None:
        """Update user login information in database."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE users SET 
                        last_login = $1,
                        failed_login_attempts = $2,
                        locked = $3,
                        locked_until = $4
                    WHERE id = $5
                """, user.last_login, user.failed_login_attempts, 
                    user.locked, user.locked_until, user.id)
        except Exception as e:
            logger.error(f"Error updating user login info: {e}")


# Global auth service instance
auth_service = None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Dependency to get current authenticated user."""
    user = await auth_service.validate_session(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return user


async def require_permission(permission: Permission):
    """Dependency factory to require specific permission."""
    async def check_permission(user: User = Depends(get_current_user)) -> User:
        if not await auth_service.authorize_action(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {permission.value} required"
            )
        return user
    return check_permission


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global db_pool, redis_client, auth_service
    
    # Initialize database connection pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    
    # Initialize Redis client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    
    # Initialize auth service
    auth_service = AuthenticationService(db_pool, redis_client)
    
    logger.info("Authentication service started")
    
    yield
    
    # Cleanup
    await db_pool.close()
    redis_client.close()
    logger.info("Authentication service stopped")


# Create FastAPI app
app = FastAPI(
    title="Project Argus Authentication Service",
    description="Authentication and Authorization API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT token."""
    user = await auth_service.authenticate_user(
        request.username, request.password, request.mfa_token
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create session token
    client_info = {"user_agent": "web", "ip": "unknown"}  # Get from request headers
    token = await auth_service.create_session(user, client_info)
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=user.to_dict()
    )


@app.post("/auth/logout")
async def logout(user: User = Depends(get_current_user), 
                credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user and revoke session."""
    await auth_service.revoke_session(credentials.credentials)
    return {"message": "Logged out successfully"}


@app.get("/auth/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current user information."""
    return user.to_dict()


@app.post("/auth/setup-mfa", response_model=MFASetupResponse)
async def setup_mfa(user: User = Depends(get_current_user)):
    """Set up multi-factor authentication for user."""
    # Generate secret
    secret = pyotp.random_base32()
    
    # Generate QR code
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user.email,
        issuer_name=MFA_ISSUER
    )
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    qr_code_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Generate backup codes
    backup_codes = [pyotp.random_base32()[:8] for _ in range(10)]
    
    # Store MFA secret in database
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET mfa_secret = $1 WHERE id = $2",
            secret, user.id
        )
    
    return MFASetupResponse(
        secret=secret,
        qr_code=f"data:image/png;base64,{qr_code_b64}",
        backup_codes=backup_codes
    )


@app.post("/auth/enable-mfa")
async def enable_mfa(mfa_token: str, user: User = Depends(get_current_user)):
    """Enable MFA after verifying setup token."""
    # Get user's MFA secret
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT mfa_secret FROM users WHERE id = $1",
            user.id
        )
        
        if not row or not row['mfa_secret']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA not set up"
            )
        
        # Verify token
        if not auth_service._verify_mfa_token(row['mfa_secret'], mfa_token):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid MFA token"
            )
        
        # Enable MFA
        await conn.execute(
            "UPDATE users SET mfa_enabled = true WHERE id = $1",
            user.id
        )
    
    await auth_service.audit_access(user.id, "enable_mfa", "system", True)
    return {"message": "MFA enabled successfully"}


@app.post("/auth/disable-mfa")
async def disable_mfa(password: str, user: User = Depends(get_current_user)):
    """Disable MFA after password verification."""
    # Verify password
    if not pwd_context.verify(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )
    
    # Disable MFA
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET mfa_enabled = false, mfa_secret = null WHERE id = $1",
            user.id
        )
    
    await auth_service.audit_access(user.id, "disable_mfa", "system", True)
    return {"message": "MFA disabled successfully"}


@app.post("/users", dependencies=[Depends(require_permission(Permission.MANAGE_USERS))])
async def create_user(request: UserCreateRequest, 
                     admin_user: User = Depends(get_current_user)):
    """Create new user (admin only)."""
    # Validate password policy
    valid, errors = await auth_service.enforce_password_policy(request.password)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Password policy violation", "errors": errors}
        )
    
    # Hash password
    password_hash = pwd_context.hash(request.password)
    
    # Create user
    try:
        async with db_pool.acquire() as conn:
            user_id = await conn.fetchval("""
                INSERT INTO users (username, email, full_name, password_hash, role, 
                                 phone_number, department, badge_number, active)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, true)
                RETURNING id
            """, request.username, request.email, request.full_name, password_hash,
                request.role, request.phone_number, request.department, request.badge_number)
            
            await auth_service.audit_access(
                admin_user.id, "create_user", user_id, True,
                {"created_user": request.username, "role": request.role}
            )
            
            return {"message": "User created successfully", "user_id": user_id}
            
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create user"
        )


@app.get("/users")
async def list_users(user: User = Depends(require_permission(Permission.VIEW_USERS))):
    """List all users."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, username, email, full_name, role, active, locked,
                       created_at, last_login, phone_number, department, badge_number
                FROM users
                ORDER BY created_at DESC
            """)
            
            users = []
            for row in rows:
                users.append({
                    "id": row['id'],
                    "username": row['username'],
                    "email": row['email'],
                    "full_name": row['full_name'],
                    "role": row['role'],
                    "active": row['active'],
                    "locked": row['locked'],
                    "created_at": row['created_at'].isoformat(),
                    "last_login": row['last_login'].isoformat() if row['last_login'] else None,
                    "phone_number": row['phone_number'],
                    "department": row['department'],
                    "badge_number": row['badge_number']
                })
            
            return {"users": users}
            
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@app.put("/users/{user_id}")
async def update_user(user_id: str, request: UserUpdateRequest,
                     admin_user: User = Depends(require_permission(Permission.MANAGE_USERS))):
    """Update user information (admin only)."""
    try:
        async with db_pool.acquire() as conn:
            # Build update query dynamically
            updates = []
            values = []
            param_count = 1
            
            for field, value in request.dict(exclude_unset=True).items():
                updates.append(f"{field} = ${param_count}")
                values.append(value)
                param_count += 1
            
            if not updates:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No fields to update"
                )
            
            values.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ${param_count}"
            
            result = await conn.execute(query, *values)
            
            if result == "UPDATE 0":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            await auth_service.audit_access(
                admin_user.id, "update_user", user_id, True,
                {"updated_fields": list(request.dict(exclude_unset=True).keys())}
            )
            
            return {"message": "User updated successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@app.post("/users/{user_id}/change-password")
async def change_password(user_id: str, request: PasswordChangeRequest,
                         current_user: User = Depends(get_current_user)):
    """Change user password."""
    # Users can only change their own password unless they're admin
    if user_id != current_user.id and not current_user.has_permission(Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only change your own password"
        )
    
    # Verify current password if changing own password
    if user_id == current_user.id:
        if not pwd_context.verify(request.current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid current password"
            )
    
    # Validate new password policy
    valid, errors = await auth_service.enforce_password_policy(request.new_password)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Password policy violation", "errors": errors}
        )
    
    # Update password
    try:
        password_hash = pwd_context.hash(request.new_password)
        
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET password_hash = $1, last_password_change = $2 WHERE id = $3",
                password_hash, datetime.utcnow(), user_id
            )
            
            await auth_service.audit_access(
                current_user.id, "change_password", user_id, True
            )
            
            return {"message": "Password changed successfully"}
            
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@app.get("/sessions")
async def get_sessions(user: User = Depends(require_permission(Permission.VIEW_AUDIT_LOGS))):
    """Get active sessions."""
    sessions = await auth_service.get_active_sessions()
    return {"sessions": sessions}


@app.delete("/sessions/{token}")
async def revoke_session(token: str, 
                        admin_user: User = Depends(require_permission(Permission.MANAGE_USERS))):
    """Revoke specific session (admin only)."""
    success = await auth_service.revoke_session(token)
    if success:
        return {"message": "Session revoked successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)