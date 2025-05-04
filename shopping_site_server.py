import hashlib
import uuid
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Request,
    Form,
    status,
    Response,
    BackgroundTasks
)
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel, Field, constr, validator
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from PIL import Image
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import requests
import json
from email.mime.text import MIMEText
import smtplib

# -------------------- 配置 --------------------
SECRET_KEY = "RxYPYcwqn54kGHai7Xbr8fV9yx27xluX0d26dn_n1dQ"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 1

# PayPal 配置 (替换为你的实际凭据)
PAYPAL_CLIENT_ID = "AStb2JacmxctJkjVV3KQp0eS3DTIRtf_IESXU7lYk0tJVppWRep4_kN1oUT8BFBIeC1YjuwvFhM55UJj"
PAYPAL_SECRET = "EMUqFhPNDtvQDHF18irAv4uSBU9vIXrfgR55pZvU4ALXm5o29tEZeTHIkQATF7DNozQmsRjAM90qEVUN"
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com"  # 使用沙盒环境，生产环境改为 api-m.paypal.com

# 数据库配置 - 添加字符集配置
DATABASE_URL = "mysql://root:LIN001524@13.215.48.147:3306/shopping_db?charset=utf8mb4"
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={'charset': 'utf8mb4'}
)
SessionLocal = sessionmaker(autocommit=False, bind=engine)

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# -------------------- 中间件 --------------------
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


# -------------------- 初始化App --------------------
app = FastAPI()
app.add_middleware(CSPMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------- 数据库模型 --------------------
class Category(Base):
    __tablename__ = "categories"
    catid = Column(Integer, primary_key=True)
    name = Column(String(100))


class Product(Base):
    __tablename__ = "products"
    pid = Column(Integer, primary_key=True)
    catid = Column(Integer)
    name = Column(String(100, collation='utf8mb4_unicode_ci'))  # 添加collation
    price = Column(Float)
    description = Column(Text(collation='utf8mb4_unicode_ci'))  # 添加collation
    image_url = Column(String(200))
    thumbnail_url = Column(String(200))


class User(Base):
    __tablename__ = "users"
    userid = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True)
    email = Column(String(100), unique=True)
    password_hash = Column(String(200))
    is_admin = Column(Boolean, default=False)


class Order(Base):
    __tablename__ = "orders"
    order_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.userid"))
    total_price = Column(Float)
    items = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    paid_at = Column(DateTime)
    status = Column(String(50), default="pending")
    transaction_id = Column(String(100))
    payment_status = Column(String(50), default="unpaid")
    digest = Column(String(64))  # SHA256 digest
    salt = Column(String(16))  # Random salt
    paypal_order_id = Column(String(50))  # PayPal order ID
    user = relationship("User", backref="orders")


Base.metadata.create_all(bind=engine)


# -------------------- Pydantic模型 --------------------
class CategoryCreate(BaseModel):
    name: constr(min_length=1, max_length=100) = Field(...)


class ProductCreate(BaseModel):
    catid: int
    name: constr(min_length=1, max_length=100) = Field(...)
    price: float = Field(..., gt=0)
    description: constr(max_length=500) = Field(default="")
    image_url: str = ""
    thumbnail_url: str = ""


class UserCreate(BaseModel):
    email: str
    password: str
    username: str = Field(..., min_length=1, max_length=100)
    is_admin: bool = False


class LoginForm(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UpdateUserInfoRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class CheckoutItem(BaseModel):
    pid: int
    quantity: int = Field(..., gt=0)


class CheckoutRequest(BaseModel):
    items: List[CheckoutItem]


# -------------------- 认证工具 --------------------
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(request: Request):
    token = request.headers.get("authorization")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        with SessionLocal() as session:
            user = session.query(User).filter(User.userid == user_id).first()
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# -------------------- PayPal API 工具 --------------------
def get_paypal_access_token():
    url = f"{PAYPAL_API_BASE}/v1/oauth2/token"
    headers = {"Accept": "application/json", "Accept-Language": "en_US"}
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data, auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET))
    response.raise_for_status()
    return response.json()["access_token"]


def create_paypal_order(order_items, total_price, currency, order_id):
    access_token = get_paypal_access_token()
    url = f"{PAYPAL_API_BASE}/v2/checkout/orders"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "amount": {
                "currency_code": currency,
                "value": str(total_price),
                "breakdown": {
                    "item_total": {"currency_code": currency, "value": str(total_price)}
                }
            },
            "items": [
                {
                    "name": f"Item {item['pid']}",
                    "sku": str(item['pid']),
                    "unit_amount": {"currency_code": currency, "value": str(item['price'])},
                    "quantity": str(item['quantity'])
                } for item in order_items
            ]
        }],
        "application_context": {
            "return_url": "http://localhost:8080/#/payment-complete",  # 改为前端路由
            "cancel_url": "http://localhost:8080/#/payment-cancel"
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    paypal_order_id = data["id"]
    approval_url = next(link["href"] for link in data["links"] if link["rel"] == "approve")
    return paypal_order_id, approval_url


def get_paypal_order_details(paypal_order_id):
    access_token = get_paypal_access_token()
    url = f"{PAYPAL_API_BASE}/v2/checkout/orders/{paypal_order_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


# -------------------- 路由 --------------------
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_name = os.path.splitext(file.filename)[0]
        image_path = f"static/images/{unique_id}_original_{file_name}{file_extension}"
        thumbnail_path = f"static/images/{unique_id}_thumbnail_{file_name}{file_extension}"

        os.makedirs("static/images", exist_ok=True)
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((100, 100))
            img.save(thumbnail_path)
        return {
            "image_url": f"/static/images/{unique_id}_original_{file_name}{file_extension}",
            "thumbnail_url": f"/static/images/{unique_id}_thumbnail_{file_name}{file_extension}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# 用户认证相关
@app.post("/api/register")
def register(user: UserCreate):
    with SessionLocal() as session:
        existing_user = session.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed_password = get_password_hash(user.password)
        new_user = User(
            email=user.email,
            password_hash=hashed_password,
            is_admin=user.is_admin,
            username=user.username
        )
        session.add(new_user)
        session.commit()
        return {"status": "success"}


@app.post("/api/login")
def login(response: Response, form_data: LoginForm):
    with SessionLocal() as session:
        user = session.query(User).filter(User.email == form_data.email).first()
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        access_token = create_access_token(
            data={"sub": str(user.userid)},
            expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
        )
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="Lax",
            max_age=86400 * ACCESS_TOKEN_EXPIRE_DAYS
        )
        return {
            "status": "success",
            "username": user.username,
            "is_admin": user.is_admin,
            "token": access_token
        }


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"status": "success"}


@app.get("/api/me")
def get_current_user_info(user: User = Depends(get_current_user)):
    return {"username": user.username, "is_admin": user.is_admin}


@app.post("/api/change-password")
def change_password(request: ChangePasswordRequest, user: User = Depends(get_current_user)):
    with SessionLocal() as session:
        if not verify_password(request.current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        user.password_hash = get_password_hash(request.new_password)
        session.commit()
    response = Response(content={"status": "success"})
    response.delete_cookie("auth_token")
    return response


@app.put("/api/update-user-info")
def update_user_info(request: UpdateUserInfoRequest, user: User = Depends(get_current_user)):
    with SessionLocal() as session:
        if request.username:
            existing_user = session.query(User).filter(User.username == request.username).first()
            if existing_user and existing_user.userid != user.userid:
                raise HTTPException(status_code=400, detail="Username already taken")
            user.username = request.username
        if request.email:
            existing_user = session.query(User).filter(User.email == request.email).first()
            if existing_user and existing_user.userid != user.userid:
                raise HTTPException(status_code=400, detail="Email already taken")
            user.email = request.email
        session.commit()
        return {"status": "success"}


# 管理员路由
@app.get("/admin")
def admin_panel(user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"message": "Welcome to Admin Panel"}


# category
@app.get("/api/categories")
def get_categories():
    with SessionLocal() as session:
        categories = session.query(Category).all()
        return [{"catid": c.catid, "name": c.name} for c in categories]


@app.post("/api/categories")
def create_category(category: CategoryCreate, user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        new_category = Category(catid=len(session.query(Category).all()) + 1, name=category.name)
        session.add(new_category)
        session.commit()
        return {"status": "success"}


@app.put("/api/categories/{catid}")
def update_category(catid: int, category: CategoryCreate, user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        existing = session.query(Category).get(catid)
        if not existing:
            raise HTTPException(status_code=404, detail="Category not found")
        existing.name = category.name
        session.commit()
        return {"status": "updated"}


@app.delete("/api/categories/{catid}")
def delete_category(catid: int, user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        category = session.query(Category).get(catid)
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        session.delete(category)
        session.commit()
        return {"status": "deleted"}


# product
@app.get("/api/products")
def get_products(catid: int = None):
    with SessionLocal() as session:
        query = session.query(Product)
        if catid:
            query = query.filter(Product.catid == catid)
        products = query.all()
        return [{
            "pid": p.pid,
            "catid": p.catid,
            "name": p.name,
            "price": p.price,
            "description": p.description,
            "image_url": p.image_url,
            "thumbnail_url": p.thumbnail_url
        } for p in products]


@app.get("/api/one_product/{pid}")
def get_product(pid: int):
    with SessionLocal() as session:
        product = session.query(Product).get(pid)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product


@app.post("/api/products")
def create_product(product: ProductCreate, user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        new_product = Product(**product.dict())
        session.add(new_product)
        session.commit()
        return {"status": "success"}


@app.put("/api/products/{pid}")
def update_product(pid: int, product: ProductCreate, user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        existing = session.query(Product).get(pid)
        if not existing:
            raise HTTPException(status_code=404, detail="Product not found")
        for key, value in product.dict().items():
            setattr(existing, key, value)
        session.commit()
        return {"status": "updated"}


@app.delete("/api/products/{pid}")
def delete_product(pid: int, user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        product = session.query(Product).get(pid)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        session.delete(product)
        session.commit()
        return {"status": "deleted"}


# order
@app.post("/api/checkout")
async def checkout(request: CheckoutRequest, user: User = Depends(get_current_user)):
    items = request.items
    with SessionLocal() as session:
        product_ids = [item.pid for item in items]
        products = session.query(Product).filter(Product.pid.in_(product_ids)).all()
        product_dict = {p.pid: p for p in products}

        total_price = 0
        order_items = []
        for item in items:
            pid = item.pid
            quantity = item.quantity
            if pid not in product_dict:
                raise HTTPException(status_code=400, detail=f"Product {pid} not found")
            price = product_dict[pid].price
            total_price += price * quantity
            order_items.append({"pid": pid, "quantity": quantity, "price": price})

        salt = secrets.token_hex(8)
        currency = "USD"
        merchant_email = "sb-xgklm39485646@business.example.com"  # 配置中替换
        digest_data = [
            currency,
            merchant_email,
            salt,
            "|".join([f"{item['pid']}:{item['quantity']}:{item['price']}" for item in order_items]),
            str(total_price)
        ]
        digest = hashlib.sha256(":".join(digest_data).encode()).hexdigest()

        new_order = Order(
            user_id=user.userid,
            total_price=total_price,
            items=json.dumps(order_items),
            status="pending",
            payment_status="unpaid",
            digest=digest,
            salt=salt
        )
        session.add(new_order)
        session.commit()

        paypal_order_id, approval_url = create_paypal_order(order_items, total_price, currency, new_order.order_id)
        new_order.paypal_order_id = paypal_order_id
        session.commit()

        return {"approval_url": approval_url}


@app.get("/api/payment-success")
async def payment_success(token: str, PayerID: str):
    try:
        with SessionLocal() as session:
            # 根据订单状态查询相关订单
            order = session.query(Order).filter(
                Order.paypal_order_id == token,
                Order.status == "pending"
            ).first()

            if not order:
                raise HTTPException(status_code=404, detail="Order not found")

            # 获取 PayPal 订单详情并验证
            access_token = get_paypal_access_token()
            url = f"{PAYPAL_API_BASE}/v2/checkout/orders/{token}/capture"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }

            # 捕获支付
            response = requests.post(url, headers=headers)
            payment_data = response.json()

            if response.status_code == 201:  # 支付成功
                order.status = "completed"
                order.payment_status = "paid"
                order.paid_at = datetime.now()
                session.commit()

                # 重定向到成功页面
                return {"status": "success", "message": "Payment processed successfully"}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Payment failed: {payment_data.get('message', 'Unknown error')}"
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paypal-webhook")
async def paypal_webhook(request: Request):
    data = await request.json()
    # 验证签名（此处简化为假设已验证）
    event_type = data.get("event_type")
    if (event_type == "PAYMENT.CAPTURE.COMPLETED"):
        paypal_order_id = data["resource"]["id"]
        with SessionLocal() as session:
            order = session.query(Order).filter(Order.paypal_order_id == paypal_order_id).first()
            if not order:
                raise HTTPException(status_code=404, detail="Order not found")
            if order.payment_status == "paid":
                return {"status": "Transaction already processed"}

            paypal_details = get_paypal_order_details(paypal_order_id)
            currency = paypal_details["purchase_units"][0]["amount"]["currency_code"]
            total_price = float(paypal_details["purchase_units"][0]["amount"]["value"])
            items = [
                {"pid": item["sku"], "quantity": int(item["quantity"]), "price": float(item["unit_amount"]["value"])}
                for item in paypal_details["purchase_units"][0]["items"]
            ]

            digest_data = [
                currency,
                "sb-xgklm39485646@business.example.com",  # 配置中替换
                order.salt,
                "|".join([f"{item['pid']}:{item['quantity']}:{item['price']}" for item in items]),
                str(total_price)
            ]
            regenerated_digest = hashlib.sha256(":".join(digest_data).encode()).hexdigest()

            if regenerated_digest != order.digest:
                raise HTTPException(status_code=400, detail="Digest validation failed")

            order.status = "completed"
            order.payment_status = "paid"
            order.transaction_id = paypal_order_id
            order.paid_at = datetime.now()
            session.commit()
            return {"status": "success"}
    return {"status": "event not handled"}


@app.get("/api/admin/orders")
def get_all_orders(user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    with SessionLocal() as session:
        orders = session.query(Order).all()
        result = []
        for o in orders:
            items = json.loads(o.items)
            # 查询每个订单项的商品信息
            for item in items:
                product = session.query(Product).get(item["pid"])
                if product:
                    item["name"] = product.name
                    item["thumbnail_url"] = product.thumbnail_url

            result.append({
                "order_id": o.order_id,
                "user_id": o.user_id,
                "total_price": o.total_price,
                "items": items,
                "status": o.status,
                "payment_status": o.payment_status,
                "created_at": o.created_at.isoformat(),
                "paid_at": o.paid_at.isoformat() if o.paid_at else None
            })
        return result


@app.get("/api/orders")
def get_user_orders(user: User = Depends(get_current_user)):
    with SessionLocal() as session:
        orders = session.query(Order).filter(
            Order.user_id == user.userid
        ).order_by(Order.created_at.desc()).all()

        result = []
        for o in orders:
            items = json.loads(o.items)
            # 查询每个订单项的商品信息
            for item in items:
                product = session.query(Product).get(item["pid"])
                if product:
                    item["name"] = product.name
                    item["thumbnail_url"] = product.thumbnail_url

            result.append({
                "order_id": o.order_id,
                "total_price": o.total_price,
                "items": items,
                "status": o.status,
                "payment_status": o.payment_status,
                "created_at": o.created_at.isoformat(),
                "paid_at": o.paid_at.isoformat() if o.paid_at else None
            })
        return result


# 邮件配置
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 25  # 465/994
SMTP_USERNAME = "colinboy0524@163.com"  # 替换为你的邮箱
SMTP_PASSWORD = "ZVbKeddm8T9LSKnQ"  # 替换为你的应用密码


# 发送邮件函数
async def send_reset_email(email: str, reset_token: str):
    print("Sending reset email to:", email)  # 添加调试信息
    reset_link = f"http://localhost:8080/#/reset-password/{reset_token}"
    message = MIMEText(
        f"Click the following link to reset your password: {reset_link}",
        "plain"
    )
    message["Subject"] = "Password Reset"
    message["From"] = SMTP_USERNAME
    message["To"] = email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            print("SMTP login successful")  # 添加调试信息
            server.send_message(message)
            print("Email sent successfully")  # 添加调试信息
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise e  # 抛出异常以便追踪问题


@app.post("/api/reset-password-request")
async def request_password_reset(background_tasks: BackgroundTasks, email: str = Form(...)):
    print(f"Received reset request for email: {email}")  # 添加调试信息
    with SessionLocal() as session:
        user = session.query(User).filter(User.email == email).first()
        if user:
            print(f"User found: {user.username}")  # 添加调试信息
            reset_token = create_access_token(
                data={"sub": str(user.userid)},
                expires_delta=timedelta(minutes=30)
            )
            try:
                # 直接调用而不是使用后台任务，方便调试
                await send_reset_email(email, reset_token)
                return {"message": "Reset email sent successfully"}
            except Exception as e:
                print(f"Error sending email: {e}")  # 添加调试信息
                raise HTTPException(status_code=500, detail=str(e))
        else:
            print("User not found")  # 添加调试信息

    return {"message": "If an account exists with this email, a reset link will be sent."}


@app.post("/api/reset-password/{token}")
async def reset_password(token: str, new_password: str = Form(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        with SessionLocal() as session:
            user = session.query(User).filter(User.userid == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            user.password_hash = get_password_hash(new_password)
            session.commit()
        return {"message": "Password reset successful"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# 启动服务
if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="/etc/letsencrypt/live/s20.iems5718.ie.cuhk.edu.hk/privkey.pem",
        ssl_certfile="/etc/letsencrypt/live/s20.iems5718.ie.cuhk.edu.hk/fullchain.pem"
    )