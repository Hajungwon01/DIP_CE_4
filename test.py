import cv2 as cv
import time
import numpy as np

# 파일 지정
Path = "../data/"  # 파일 경로
Name = 'frozen.avi'  # 파일 이름

FullName = Path + Name
SaveFileName = '4조.avi'  # 저장할 파일 이름

videoCapture = cv.VideoCapture(FullName)  # 읽기용 객체 생성

# 동영상 파일이 제대로 열렸는지 확인
if not videoCapture.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()  # 동영상 파일을 열 수 없다면 프로그램 종료

# 동영상의 정보
number_of_total_frames = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
fps = videoCapture.get(cv.CAP_PROP_FPS)
dly_ms = 1000 / (fps)

# 창 만들기
cv.namedWindow("Video Player : Team 4")

# resize 기능을 위한 width와 height 정의
resize_height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT) // 2)
resize_width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH) // 2)

success, frame = videoCapture.read()  # 동영상을 성공적으로 열었을 경우 프레임을 받아온다

count = 0
margin = 1  # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

is_paused = False  # 동영상 일시 정지 상태 변수

s_time = time.time()  # ms 단위의 현재 tick count을 반환

# 최대 프레임 인덱스
max_frame_index = int(number_of_total_frames) - 1
current_frame_index = 0

# 트랙바 콜백 함수
def slideCallBack(pos):
    global current_frame_index, count
    current_frame_index = pos
    count = pos
    videoCapture.set(cv.CAP_PROP_POS_FRAMES, current_frame_index)

# 트랙바 생성 및 콜백 함수 연결
cv.createTrackbar('SLIDE', 'Video Player : Team 4', 0, max_frame_index, slideCallBack)

# 초기 밝기 설정
brightness = 10

# 트랙바 생성 및 콜백 함수 연결
cv.createTrackbar('Brightness', 'Video Player : Team 4', brightness, 20, lambda x: None)

while success:  # Loop until there are no more frames.
    s = time.time()  # start. time in sec.

    resize_frame = cv.resize(frame.copy(), (resize_width, resize_height))  # resize 기능을 적용한 frame
    original = resize_frame.copy()  # 원본 영상
    frame_scaling = cv.convertScaleAbs(resize_frame.copy(), alpha=brightness / 10.0)  # scaling된 영상

    # 현재 프레임 인덱스를 원본 영상과 처리된 영상의 좌측 상단에 빨간색으로 표시
    cv.putText(original, f'original: {count}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv.putText(frame_scaling, f'scaling: {count}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    # 화면 분할 기능 : 원본 영상과 scaling된 영상을 x축(가로 방향)상으로 이어붙이기
    new_frame = np.concatenate((original, frame_scaling), axis=1)

    cv.imshow('Video Player : Team 4', new_frame)

    # 재생중인 영상의 프레임 인덱스와 트랙바 위치를 업데이트
    current_frame_index += 1
    count += 1
    cv.setTrackbarPos('SLIDE', 'Video Player : Team 4', current_frame_index)

    key = cv.waitKey(1)
    if key == 27:  # esc 키를 누르면 비디오 종료
        break
    elif key == ord(' '):  # 스페이스바를 누르면 동영상 일시 정지/재개
        is_paused = not is_paused
        while is_paused:
            key2 = cv.waitKey(1)
            if key2 == ord(' '):  # 스페이스바를 다시 누르면 동영상 일시 정지 해제
                is_paused = False

    # 밝기 트랙바 값 가져오기
    brightness = cv.getTrackbarPos('Brightness', 'Video Player : Team 4')

    success, frame = videoCapture.read()  # 다음 프레임을 읽어온다.
    print("\rCurrent frame number = ", count, end=' ')
    while ((time.time() - s) * 1000) < (dly_ms - margin):  # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

e_time = time.time() - s_time
playing_sec = number_of_total_frames / fps  # 상영시간[sec]
print(f'\n\nExpected play time={playing_sec:#.2f}[sec]')  # 수행시간 출력 기능
print(f'Real play time={e_time:#.2f}[sec]')

videoCapture.release()
