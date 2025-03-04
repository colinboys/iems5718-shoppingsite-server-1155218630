# main.py
import uuid

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from PIL import Image
import os
from starlette.middleware.cors import CORSMiddleware




app = FastAPI()
# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()

engine = create_engine("mysql+pymysql://admin:LIN001524@database-1.cdu8y6esawv.ap-southeast-1.rds.amazonaws.com:3306/shopping_db")

SessionLocal = sessionmaker(autocommit=False, bind=engine)

# static file
app.mount("/static", StaticFiles(directory="static"), name="static")


# database model
class Category(Base):
    __tablename__ = "categories"
    catid = Column(Integer, primary_key=True)
    name = Column(String)


class Product(Base):
    __tablename__ = "products"
    pid = Column(Integer, primary_key=True)
    catid = Column(Integer)
    name = Column(String)
    price = Column(Float)
    description = Column(Text)
    image_url = Column(String)
    thumbnail_url = Column(String)

class CategoryCreate(BaseModel):
    name: str

class ProductCreate(BaseModel):
    catid: int
    name: str
    price: float
    description: str = ""
    image_url: str = ""
    thumbnail_url: str = ""


Base.metadata.create_all(bind=engine)








# Pydantic model
class CategoryCreate(BaseModel):
    name: str

class ProductCreate(BaseModel):
    catid: int
    name: str
    price: float
    description: str = ""
    image_url: str = ""
    thumbnail_url: str = ""


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # unique id generated
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_name = os.path.splitext(file.filename)[0]
        image_path = f"static/images/{unique_id}_original_{file_name}{file_extension}"
        thumbnail_path = f"static/images/{unique_id}_thumbnail_{file_name}{file_extension}"

        os.makedirs("static/images", exist_ok=True)

        # save original picture
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # generate thumbnail
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")  # convert to RGB mode
            img.thumbnail((100, 100))
            img.save(thumbnail_path)

        return {
            "image_url": f"/static/images/{unique_id}_original_{file_name}{file_extension}",
            "thumbnail_url": f"/static/images/{unique_id}_thumbnail_{file_name}{file_extension}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片上传失败: {str(e)}")

# category
@app.get("/api/categories")
def get_categories():
    with SessionLocal() as session:
        categories = session.query(Category).all()
        for c in categories:
            print(c.name)
        return [{"catid": c.catid, "name": c.name} for c in categories]

@app.post("/api/categories")
def create_category(category: CategoryCreate):
    with SessionLocal() as session:
        new_category = Category(catid=len(session.query(Category).all())+1,name=category.name)
        session.add(new_category)
        session.commit()
        return {"status": "success"}

@app.put("/api/categories/{catid}")
def update_category(catid: int, category: CategoryCreate):
    with SessionLocal() as session:
        existing = session.query(Category).get(catid)
        if not existing:
            raise HTTPException(status_code=404, detail="Category not found")
        existing.name = category.name
        session.commit()
        return {"status": "updated"}

@app.delete("/api/categories/{catid}")
def delete_category(catid: int):
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
        for p in products:
            print(p.name)
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
def get_products(pid: int):
    with SessionLocal() as session:

        product = session.query(Product).get(pid)
        return product

@app.post("/api/products")
def create_product(product: ProductCreate):
    with SessionLocal() as session:
        new_product = Product(**product.dict())
        session.add(new_product)
        session.commit()
        return {"status": "success"}

@app.put("/api/products/{pid}")
def update_product(pid: int, product: ProductCreate):
    with SessionLocal() as session:
        existing = session.query(Product).get(pid)
        if not existing:
            raise HTTPException(status_code=404, detail="Product not found")
        for key, value in product.dict().items():
            setattr(existing, key, value)
        session.commit()
        return {"status": "updated"}

@app.delete("/api/products/{pid}")
def delete_product(pid: int):
    with SessionLocal() as session:
        product = session.query(Product).get(pid)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        session.delete(product)
        session.commit()
        return {"status": "deleted"}


# launch Uvicorn service
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
