from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from models import pix2pix_model, networks
from torchvision import transforms
import argparse
import base64, cv2
import numpy as np
import torch
from PIL import Image
import io

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

opt = argparse.Namespace(
    input_nc=3,
    output_nc=3,
    ngf=64,
    netG='unet_256',
    norm='instance',
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    # gpu_ids=[0],
    gpu_ids=[],
    activation='swish',
    squeeze=4
)

netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, opt.use_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, opt.activation, opt.squeeze)
# netG = netG.module
path = './300_net_G.pth'
order_dict = torch.load(path)
netG.load_state_dict(order_dict)
netG.eval()
netG = netG.to(device)

w,h = 720, 480 
transform = transforms.Compose([transforms.CenterCrop((h//2,w//2)),transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

app = FastAPI()
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
import random
import string
import time
import asyncio, aiohttp
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc import RTCSessionDescription, RTCPeerConnection, VideoStreamTrack
import math
from av import VideoFrame

user_video_frame = []


class FlagVideoStreamTrack(VideoStreamTrack):
	"""
    A video track that returns an animated flag.
    """

	def __init__(self):
		super().__init__()  # don't forget this!
		self.counter = 0
		height, width = 480, 640

		# generate flag
		data_bgr = np.hstack(
			[
				self._create_rectangle(
					width=213, height=480, color=(255, 0, 0)
				),  # blue
				self._create_rectangle(
					width=214, height=480, color=(255, 255, 255)
				),  # white
				self._create_rectangle(width=213, height=480, color=(0, 0, 255)),  # red
			]
		)

		# shrink and center it
		M = np.float32([[0.5, 0, width / 4], [0, 0.5, height / 4]])
		data_bgr = cv2.warpAffine(data_bgr, M, (width, height))

		# compute animation
		omega = 2 * math.pi / height
		id_x = np.tile(np.array(range(width), dtype=np.float32), (height, 1))
		id_y = np.tile(
			np.array(range(height), dtype=np.float32), (width, 1)
		).transpose()

		self.frames = []
		for k in range(30):
			phase = 2 * k * math.pi / 30
			map_x = id_x + 10 * np.cos(omega * id_x + phase)
			map_y = id_y + 10 * np.sin(omega * id_x + phase)
			self.frames.append(
				VideoFrame.from_ndarray(
					cv2.remap(data_bgr, map_x, map_y, cv2.INTER_LINEAR), format="bgr24"
				)
			)

	async def recv(self):
		pts, time_base = await self.next_timestamp()
		frame = self.frames[self.counter % 30]
		if (len(user_video_frame) > 0):
			frame = user_video_frame[0]
		frame.pts = pts
		frame.time_base = time_base
		self.counter += 1
		return frame

	def _create_rectangle(self, width, height, color):
		data_bgr = np.zeros((height, width, 3), np.uint8)
		data_bgr[:, :] = color
		return data_bgr


pcs = set()


def transaction_id():
	return "".join(random.choice(string.ascii_letters) for x in range(12))


class JanusPlugin:
	def __init__(self, session, url):
		self._queue = asyncio.Queue()
		self._session = session
		self._url = url

	async def send(self, payload):
		message = {"janus": "message", "transaction": transaction_id()}
		message.update(payload)
		async with self._session._http.post(self._url, json=message) as response:
			data = await response.json()
			assert data["janus"] == "ack"

		response = await self._queue.get()
		assert response["transaction"] == message["transaction"]
		return response


class JanusSession:
	def __init__(self, url):
		self._http = None
		self._poll_task = None
		self._plugins = {}
		self._root_url = url
		self._session_url = None

	async def attach(self, plugin_name: str) -> JanusPlugin:
		message = {
			"janus": "attach",
			"plugin": plugin_name,
			"transaction": transaction_id(),
		}
		async with self._http.post(self._session_url, json=message) as response:
			data = await response.json()
			assert data["janus"] == "success"
			plugin_id = data["data"]["id"]
			plugin = JanusPlugin(self, self._session_url + "/" + str(plugin_id))
			self._plugins[plugin_id] = plugin
			return plugin

	async def create(self):
		self._http = aiohttp.ClientSession()
		message = {"janus": "create", "transaction": transaction_id()}
		async with self._http.post(self._root_url, json=message) as response:
			data = await response.json()
			assert data["janus"] == "success"
			session_id = data["data"]["id"]
			self._session_url = self._root_url + "/" + str(session_id)

		self._poll_task = asyncio.ensure_future(self._poll())

	async def destroy(self):
		if self._poll_task:
			self._poll_task.cancel()
			self._poll_task = None

		if self._session_url:
			message = {"janus": "destroy", "transaction": transaction_id()}
			async with self._http.post(self._session_url, json=message) as response:
				data = await response.json()
				assert data["janus"] == "success"
			self._session_url = None

		if self._http:
			await self._http.close()
			self._http = None

	async def _poll(self):
		while True:
			params = {"maxev": 1, "rid": int(time.time() * 1000)}
			async with self._http.get(self._session_url, params=params) as response:
				data = await response.json()
				if data["janus"] == "event":
					plugin = self._plugins.get(data["sender"], None)
					if plugin:
						await plugin._queue.put(data)
					else:
						print(data)


async def publish(plugin, player):
	"""
    Send video to the room.
    """
	pc = RTCPeerConnection()
	pcs.add(pc)

	# configure media
	media = {"audio": False, "video": True}
	if player and player.audio:
		pc.addTrack(player.audio)
		media["audio"] = True

	if player and player.video:
		# pc.addTrack(player.video)
		pc.addTrack(FlagVideoStreamTrack())
	else:
		pc.addTrack(VideoStreamTrack())

	# send offer
	await pc.setLocalDescription(await pc.createOffer())
	request = {"request": "configure"}
	request.update(media)
	response = await plugin.send(
		{
			"body": request,
			"jsep": {
				"sdp": pc.localDescription.sdp,
				"trickle": False,
				"type": pc.localDescription.type,
			},
		}
	)

	# apply answer
	await pc.setRemoteDescription(
		RTCSessionDescription(
			sdp=response["jsep"]["sdp"], type=response["jsep"]["type"]
		)
	)


async def subscribe(session, room, feed, recorder):
	pc = RTCPeerConnection()
	pcs.add(pc)

	@pc.on("track")
	async def on_track(track):
		print("Track %s received" % track.kind)
		if track.kind == "video":
			recorder.addTrack(track)
		if track.kind == "audio":
			recorder.addTrack(track)

	# subscribe
	plugin = await session.attach("janus.plugin.videoroom")
	response = await plugin.send(
		{"body": {"request": "join", "ptype": "subscriber", "room": room, "feed": feed}}
	)

	# apply offer
	await pc.setRemoteDescription(
		RTCSessionDescription(
			sdp=response["jsep"]["sdp"], type=response["jsep"]["type"]
		)
	)

	# send answer
	await pc.setLocalDescription(await pc.createAnswer())
	response = await plugin.send(
		{
			"body": {"request": "start"},
			"jsep": {
				"sdp": pc.localDescription.sdp,
				"trickle": False,
				"type": pc.localDescription.type,
			},
		}
	)
	await recorder.start()


async def run(session):
	await session.create()
	room = 1234
	# join video room
	player = MediaPlayer('default:none', format='avfoundation', options={
		"video_size": "640x480",
		"framerate": "30",  # 30fps로 변경
		"pixel_format": "uyvy422",  # 사용 가능한 픽셀 포맷 중 하나를 선택하세요
		"input_device": "Capture screen 0",  # 화면 캡처 장치의 이름을 여기에 입력
	})
	recorder = None
	plugin = await session.attach("janus.plugin.videoroom")
	response = await plugin.send(
		{
			"body": {
				"display": "aiortc",
				"ptype": "publisher",
				"request": "join",
				"room": room,
			}
		}
	)
	publishers = response["plugindata"]["data"]["publishers"]
	for publisher in publishers:
		print("id: %(id)s, display: %(display)s" % publisher)

	# send video
	await publish(plugin=plugin, player=player)

	# receive video
	if recorder is not None and publishers:
		await subscribe(
			session=session, room=room, feed=publishers[0]["id"], recorder=recorder
		)

	# exchange media for 10 minutes
	print("Exchanging media")
	await asyncio.sleep(600)


@app.get("/")
async def root():
	base_url = "http://localhost:8088/janus"
	session = JanusSession(
		url=base_url,
	)
	await run(session=session)
	return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
	return {"message": f"Hello {name}"}


@socket_manager.on('image')
async def handle_join(sid, *args, **kwargs):
	# print("image")
	# print(args[0])
	image_data = base64.b64decode(args[0].split(',')[1])
	image = Image.open(io.BytesIO(image_data))
	t_img = transform(image)
	t_img = t_img.view(1, 3, 512, 512)
	t_img = t_img.to(device)
	with torch.no_grad():
		out = netG(t_img)
	out = out.to('cpu')
	out = out * 0.5 + 0.5
	image_np = out[0].numpy()
	image_np = np.transpose(image_np, (1, 2, 0))
	image_np = (image_np * 255).astype(np.uint8)
	if (len(user_video_frame) == 0):
		user_video_frame.append([])
	user_video_frame[0] = (
		VideoFrame.from_ndarray(
			image_np
			# cv2.remap(image_np, 320, 320, cv2.INTER_LINEAR), format="bgr24"
		)
	)
	opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
	frame_resized = cv2.resize(opencv_image, (640, 360))
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
	processed_img_data = base64.b64encode(frame_encoded).decode()
	b64_src = "data:image/jpg;base64,"
	processed_img_data = b64_src + processed_img_data
	print(user_video_frame)
	await app.sio.emit("processed_image", processed_img_data)
