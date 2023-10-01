"""
제출자 성명: 하정원, 정성원, 박해윤
제출자 학과 및 학번: 컴퓨터공학과 2020305082, 컴퓨터공학과 2020305068,


1차 과제 - 동영상 플레이어

    1. 플레이 기능:
       한 화면 = 좌측(원본 영상) + 우측(밝기를 조절한 영상)
        - 화면이 너무 커지지 않도록 좌측과 우측 영상은 각각 1/2로 줄여서 사용하기 바랍니다.    playing - 기본. 항상 진행
        forwarding, backwarding - Trackbar로 제어
        scaling - Trackbar로 0~20까지. 10이 시작점. 10이하는 어둡게, 10이상은 밝게..
        정지 - 스페이스 바
        종료 - esc 키

    2. 재생 중 정보 처리 기능
        현재 처리 중인 frame의 인덱스 번호를 원본과 처리된 영상의 좌측 상단에 출력

    3. 저장 기능
         재생이 완료되면 자동으로  파일이 저장되어야 한다. 파일이름: 1조.avi  등

    4. 수행시간 출력 기능
        실행창에는 이론적 재생시간과 실제 재생시간이 표현되어야 한다.(그 차이가 적을 수록 유리)

"""

# 필요한 모듈 import
import cv2 as cv
import time
import numpy as np

# 파일 지정
Path = "../data/" # 파일 경로
Name = 'frozen.avi' # 파일 이름

FullName = Path + Name
SaveFileName = '4조.avi' # 저장할 파일 이름

videoCapture = cv.VideoCapture(FullName) # 읽기용 객체 생성

# 동영상 파일이 제대로 열렸는지 확인
if not videoCapture.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit() # 동영상 파일을 열 수 없다면 프로그램 종료

# 동영상의 정보
number_of_total_frames = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
fps = videoCapture.get(cv.CAP_PROP_FPS)
dly_ms = 1000/(fps)

# 창 만들기
# cv.namedWindow("Video Player : Team 4")

# resize 기능을 위한 width와 height 정의
resize_height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)//2)
resize_width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)//2)

success, frame = videoCapture.read() # 동영상을 성공적으로 열었을 경우 프레임을 받아온다

count = 0
margin = 1      # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

is_paused = False  # 동영상 일시 정지 상태 변수

s_time = time.time()            # ms 단위의 현재 tick count을 반환
while success:          # Loop until there are no more frames.
    s = time.time()     # start. time in sec.

    resize_frame = cv.resize(frame.copy(), (resize_width, resize_height)) # resize 기능을 적용한 frame
    original = resize_frame.copy() # 원본 영상
    frame_scaling = resize_frame.copy() # scaling된 영상

    new_frame = np.concatenate((original, frame_scaling), axis=1) # 화면 분할 기능 : 원본 영상과 scaling된 영상을 x축(가로 방향)상으로 이어붙이기

    cv.imshow('Video Player : 4조'.encode('utf-8').decode(), new_frame)

    count += 1

    key = cv.waitKey(1)
    if key == 27:  # esc 키를 누르면 비디오 종료
        break
    elif key == ord(' '):  # 스페이스바를 누르면 동영상 일시 정지/재개
        is_paused = not is_paused
        while is_paused:
            key2 = cv.waitKey(1)
            if key2 == ord(' '):  # 스페이스바를 다시 누르면 동영상 일시 정지 해제
                is_paused = False

    success, frame = videoCapture.read()    # 다음 프레임을 읽어온다.
    print("\rCurrent frame number = ", count, end=' ')
    while ( (time.time() - s) * 1000 ) < (dly_ms - margin): # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass
e_time = time.time() - s_time
playing_sec = number_of_total_frames/fps  # 상영시간[sec]
print(f'\n\nExpected play time={playing_sec:#.2f}[sec]') # 수행시간 출력 기능
print(f'Real play time={e_time:#.2f}[sec]')

videoCapture.release()