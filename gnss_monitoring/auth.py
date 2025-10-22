"""
用户认证和授权模块

本模块提供API安全功能：
- JWT token认证
- 用户管理
- 角色和权限
- API密钥管理
- 密码加密
- 会话管理

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import secrets
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    class BaseModel:
        pass

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)


class AuthError(CHSBDSException):
    """认证异常"""
    pass


class Role(str, Enum):
    """用户角色"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class Permission(str, Enum):
    """权限"""
    READ_DATA = "read:data"
    WRITE_DATA = "write:data"
    DELETE_DATA = "delete:data"
    RUN_ANALYSIS = "run:analysis"
    MANAGE_USERS = "manage:users"
    MANAGE_SYSTEM = "manage:system"
    VIEW_ALERTS = "view:alerts"
    MANAGE_ALERTS = "manage:alerts"


# 角色权限映射
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.DELETE_DATA,
        Permission.RUN_ANALYSIS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SYSTEM,
        Permission.VIEW_ALERTS,
        Permission.MANAGE_ALERTS,
    ],
    Role.OPERATOR: [
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ALERTS,
        Permission.MANAGE_ALERTS,
    ],
    Role.VIEWER: [
        Permission.READ_DATA,
        Permission.VIEW_ALERTS,
    ],
    Role.API_CLIENT: [
        Permission.READ_DATA,
        Permission.RUN_ANALYSIS,
    ]
}


# Pydantic Models

class Token(BaseModel):
    """Token响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token数据"""
    username: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []


class User(BaseModel):
    """用户模型"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: List[str] = []
    disabled: bool = False
    created_at: Optional[datetime] = None


class UserInDB(User):
    """数据库中的用户（包含密码哈希）"""
    hashed_password: str


class UserCreate(BaseModel):
    """创建用户请求"""
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: List[str] = [Role.VIEWER.value]


class APIKey(BaseModel):
    """API密钥"""
    key: str
    name: str
    user_id: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


class PasswordHasher:
    """密码加密器"""

    def __init__(self):
        if not PASSLIB_AVAILABLE:
            raise AuthError("passlib is required. Install with: pip install passlib[bcrypt]")

        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(self, password: str) -> str:
        """加密密码"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self.pwd_context.verify(plain_password, hashed_password)


class JWTManager:
    """JWT token管理器"""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30
    ):
        if not JWT_AVAILABLE:
            raise AuthError("python-jose is required. Install with: pip install python-jose[cryptography]")

        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def decode_token(self, token: str) -> Dict[str, Any]:
        """解码token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise AuthError(f"Invalid token: {e}")

    def verify_token(self, token: str) -> TokenData:
        """验证token并返回token数据"""
        try:
            payload = self.decode_token(token)
            username: str = payload.get("sub")
            roles: List[str] = payload.get("roles", [])
            permissions: List[str] = payload.get("permissions", [])

            if username is None:
                raise AuthError("Invalid token payload")

            return TokenData(
                username=username,
                roles=roles,
                permissions=permissions
            )
        except JWTError:
            raise AuthError("Could not validate credentials")


class UserManager:
    """用户管理器"""

    def __init__(self):
        self.password_hasher = PasswordHasher() if PASSLIB_AVAILABLE else None

        # 内存存储（生产环境应使用数据库）
        self.users_db: Dict[str, UserInDB] = {}

        # 创建默认管理员用户
        self._create_default_admin()

    def _create_default_admin(self):
        """创建默认管理员账户"""
        if 'admin' not in self.users_db and self.password_hasher:
            admin_user = UserInDB(
                username='admin',
                email='admin@chsbds.local',
                full_name='Administrator',
                roles=[Role.ADMIN.value],
                hashed_password=self.password_hasher.hash_password('admin123'),
                disabled=False,
                created_at=datetime.now()
            )
            self.users_db['admin'] = admin_user
            logger.info("Default admin user created (username: admin, password: admin123)")

    def create_user(self, user_create: UserCreate) -> User:
        """创建新用户"""
        if user_create.username in self.users_db:
            raise AuthError(f"User {user_create.username} already exists")

        if not self.password_hasher:
            raise AuthError("Password hasher not available")

        hashed_password = self.password_hasher.hash_password(user_create.password)

        user = UserInDB(
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            roles=user_create.roles,
            hashed_password=hashed_password,
            disabled=False,
            created_at=datetime.now()
        )

        self.users_db[user_create.username] = user

        logger.info(f"User created: {user_create.username}")

        # 返回不包含密码的用户对象
        return User(**user.dict(exclude={'hashed_password'}))

    def get_user(self, username: str) -> Optional[UserInDB]:
        """获取用户"""
        return self.users_db.get(username)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """认证用户"""
        user = self.get_user(username)

        if not user:
            return None

        if not self.password_hasher:
            return None

        if not self.password_hasher.verify_password(password, user.hashed_password):
            return None

        if user.disabled:
            return None

        return User(**user.dict(exclude={'hashed_password'}))

    def get_user_permissions(self, user: User) -> List[str]:
        """获取用户权限"""
        permissions = set()

        for role_str in user.roles:
            try:
                role = Role(role_str)
                permissions.update(ROLE_PERMISSIONS.get(role, []))
            except ValueError:
                logger.warning(f"Unknown role: {role_str}")

        return list(permissions)


class AuthenticationService:
    """认证服务"""

    def __init__(self):
        self.jwt_manager = JWTManager() if JWT_AVAILABLE else None
        self.user_manager = UserManager()

    def login(self, username: str, password: str) -> Token:
        """用户登录"""
        if not self.jwt_manager:
            raise AuthError("JWT manager not available")

        # 认证用户
        user = self.user_manager.authenticate_user(username, password)
        if not user:
            raise AuthError("Incorrect username or password")

        # 获取权限
        permissions = self.user_manager.get_user_permissions(user)

        # 创建token
        access_token = self.jwt_manager.create_access_token(
            data={
                "sub": user.username,
                "roles": user.roles,
                "permissions": permissions
            }
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.jwt_manager.access_token_expire_minutes * 60
        )

    def verify_token(self, token: str) -> TokenData:
        """验证token"""
        if not self.jwt_manager:
            raise AuthError("JWT manager not available")

        return self.jwt_manager.verify_token(token)

    def get_current_user(self, token: str) -> User:
        """从token获取当前用户"""
        token_data = self.verify_token(token)
        user = self.user_manager.get_user(token_data.username)

        if user is None:
            raise AuthError("User not found")

        return User(**user.dict(exclude={'hashed_password'}))


# FastAPI依赖

if FASTAPI_AVAILABLE:
    security = HTTPBearer()

    def get_auth_service() -> AuthenticationService:
        """获取认证服务（FastAPI依赖）"""
        return AuthenticationService()

    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_service: AuthenticationService = Depends(get_auth_service)
    ) -> User:
        """获取当前用户（FastAPI依赖）"""
        try:
            token = credentials.credentials
            user = auth_service.get_current_user(token)
            return user
        except AuthError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def require_permission(permission: Permission):
        """要求特定权限的依赖"""
        async def permission_checker(current_user: User = Depends(get_current_user)):
            user_permissions = UserManager().get_user_permissions(current_user)

            if permission.value not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value} required"
                )

            return current_user

        return permission_checker

    async def require_role(role: Role):
        """要求特定角色的依赖"""
        async def role_checker(current_user: User = Depends(get_current_user)):
            if role.value not in current_user.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}"
                )

            return current_user

        return role_checker


# 便捷函数

def create_auth_service() -> AuthenticationService:
    """创建认证服务实例"""
    return AuthenticationService()


if __name__ == "__main__":
    print("Authentication module")

    required_packages = []
    if not PASSLIB_AVAILABLE:
        required_packages.append("passlib[bcrypt]")
    if not JWT_AVAILABLE:
        required_packages.append("python-jose[cryptography]")

    if required_packages:
        print("\nWarning: Missing dependencies")
        print(f"Install with: pip install {' '.join(required_packages)}")
    else:
        print("\nAll dependencies available")

        # 测试
        auth_service = create_auth_service()
        print("\nDefault admin user created:")
        print("  Username: admin")
        print("  Password: admin123")
