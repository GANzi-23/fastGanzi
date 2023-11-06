# model.py
from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
from db import Base
from db import ENGINE

class UserTable(Base): # 유저 테이블
    __tablename__ = 'user' #테이블 이름
    id = Column(Integer, primary_key=True, autoincrement=True) # 숫자 형식의 index 번호 자동 생성
    user_name = Column(String(50), nullable=False) # 스트링 형식의 널 삽입 불가 컬럼
    age = Column(Integer) # 숫자 컬럼

class User(BaseModel): # 유저
    id: int
    user_name: str
    age: int