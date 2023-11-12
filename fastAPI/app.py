
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from pydantic import BaseModel




# app config
app = FastAPI(title="Ganzi API",
              description="API description",
              version="0.0.0",
              docs_url="/api",
              redoc_url="/redoc")

socket_manager = SocketManager(app=app)

origins = [
	"http://localhost",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


# basemodel

class SignUpForm(BaseModel):
	username: str
	email: str
	password: str


class LoginForm(BaseModel):
	email: str
	password: str


class ModelChangeForm(BaseModel):
	room: str
	id: str
	model: str

# method
@app.get("/")
def root():
	return {"message": "Hello World"}


@app.post("/signup", status_code=201)
async def signup(form: SignUpForm):
	return {'message': "회원가입 성공", 'username': form.username}


@app.post("/login", status_code=200)
async def signup(form: LoginForm):
	return {'message': "로그인 성공", 'username': form.username}


@app.post("/model", status_code=200)
async def changemodel(form: ModelChangeForm):
	return {'message': "모델 변환 성공", 'model': form.model}

email_to_socket_mapping = dict()
socket_to_email_mapping = dict()


@socket_manager.on('join-room')
async def join_room(sid, *args, **kwargs):
	data_dict = args[0]  # 튜플 안의 딕셔너리를 추출
	emailId = data_dict['emailId']
	roomId = data_dict['roomId']
	print(f'sid = {sid} emailId = {emailId}, roomId = {roomId}')
	email_to_socket_mapping[emailId] = sid
	socket_to_email_mapping[sid] = emailId
	await socket_manager.enter_room(sid, roomId)
	await socket_manager.emit('joined-room', {'roomId': roomId}, sid)
	await socket_manager.emit("user-joined", {'emailId': emailId}, roomId)


# socket_manager.

@socket_manager.on('call-user')
async def call_user(sid, *args, **kwargs):
	data_dict = args[0]
	print(f"call_user event data =  {data_dict}")
	emailId = data_dict['emailId']
	offer = data_dict['offer']
	from_email = socket_to_email_mapping.get(sid)
	socket_id = email_to_socket_mapping.get(emailId)
	print(f"from {from_email} socket id = {socket_id}")
	await socket_manager.emit('incomming-call', {'from': from_email, 'offer': offer}, socket_id)


@socket_manager.on('call-accepted')
async def call_accepted(sid, *args, **kwargs):
	data_dict = args[0]
	print(f"call_accepted event data =  {data_dict}")
	emailId = data_dict['emailId']
	ans = data_dict['ans']
	socket_id = email_to_socket_mapping.get(emailId)
	print(f"emailId =  {emailId} socket id = {socket_id} ans = {ans}")
	await socket_manager.emit('call-accepted', {'ans': ans}, socket_id)
