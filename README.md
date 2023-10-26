# fastGanzi
Ganzi-fastAPI

```
git clone git@github.com:GANzi-23/fastGanzi
```

## Janus 서버, Janus 클라이언트 실행

```
docker compose up -d  ## -> 좀 오래걸림
```
localhost 에 접속하면 Demo Page 가 나오고 Demos -> Videoroom -> start -> 이름 입력(alphanumeric) -> 카메라 허용

(가상환경 만들고 하세요)
## Backend 서버 실행 

### 간단히 변환만 테스트 

1. pip install -r requirements.txt
2. uvicorn app:app --host localhost --port 8000 --reload
3. videosender.html 경로 복사 후 브라우저에 붙여넣기 (setInterval(sendImage,3000); 3000 을 수정하여 원하는 속도로 서버에 카메라 이미지를 보낼 수 있음 ms 단위)
4. 비디오 허용 후 카메라 화면 밑에 있는 send 누르기
5. 카메라 화면 밑에 변환된 화면 뜸

### Janus 서버에 Publish 까지 테스트 

- 위 과정을 진행
- localhost:8000 브라우저 창에 입력
