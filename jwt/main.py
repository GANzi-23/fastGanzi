from logging import error
import jwt
from datetime import timedelta, datetime
from fastapi import Depends, FastAPI
from typing import List
from starlette.middleware.cors import CORSMiddleware

from db import session
from model import UserTable, User
from fastapi.security import OAuth2PasswordBearer

# OAuth2PasswordBearer 객체를 생성할 때 tokenUrl이라는 파라미터를 넘겨준다. 프론트엔드에서 token 값을 얻어 올 때 사용한다.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/users")
async def read_users():
    users = session.query(UserTable).all()
    return users

def encode_token(username):
    if username :
        payload = {
            'exp': datetime.utcnow() + timedelta(weeks=5),
            'iat': datetime.utcnow(),
            'scope': 'access_token',
            'data': username
        }
        return jwt.encode(
            payload,
            'test',
            algorithm='HS256'
        )
    else :
        return error

@app.get("/users/{user_name}")
async def read_user(user_name):
    user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
    if user :
        return encode_token(user.user_name)
    else :
        return user

@app.post("/user")
async def create_users(user_name: str, age: int):
    user = UserTable()
    user.user_name = user_name
    user.age = age

    session.add(user)
    session.commit()

    return f"{user_name} created"

@app.put("/users")
async def update_user(users: List[User]):
    for i in users:
        user = session.query(UserTable).filter(UserTable.id == i.id).first()
        user.user_name = i.user_name
        user.age = i.age
        session.commit()

    return f"{users[0].user_name} updated"

@app.delete("/user")
async def delete_users(user_id: int):
    user = session.query(UserTable).filter(UserTable.id == user_id).delete()
    session.commit()

    return read_users

