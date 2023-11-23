"""
제출자 성명: 하정원, 정성원, 박해윤
제출자 학과 및 학번: 컴퓨터공학과 2020305082, 컴퓨터공학과 2020305068, 컴퓨터공학과 2018305028

"""

# 필요한 모듈 import
import cv2 as cv
import time
import numpy as np
import os
import copy


# position 트랙바 콜백 함수
def position_callback(pos):
    videoCapture.set(cv.CAP_PROP_POS_FRAMES, pos)


# ======================================================

#AOI 선택을 위한 전역 변수
drawing = False
roi_start = (0, 0)
roi_end = (0, 0)
original_copy = None

# ======================================================

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global drawing, roi_start, roi_end, original_copy

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        original_copy = original.copy()

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)

# ======================================================

# 파일 지정
Path = "../data/"  # 파일 경로
Name = 'frozen.avi'  # 파일 이름

FullName = Path + Name
SaveFileName = '4조.avi'  # 저장할 파일 이름

# ======================================================

videoCapture = cv.VideoCapture(FullName)  # 읽기용 객체 생성

# ======================================================

# 동영상 파일이 제대로 열렸는지 확인
if not videoCapture.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()  # 동영상 파일을 열 수 없다면 프로그램 종료

# ======================================================

# 동영상의 정보
number_of_total_frames = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
fps = videoCapture.get(cv.CAP_PROP_FPS)
dly_ms = 1000/(fps)

# 동영상의 너비와 높이
width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))

# 가로로 이어 붙인 화면의 크기가 FHD 해상도(1920x1080) 이내인지 확인
resize_needed = (width*2 > 1920) or (height > 1080)

# resize 기능을 위한 width와 height 정의
resize_height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT) // 2)
resize_width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH) // 2)

# 최대 프레임 인덱스
max_frame_index = int(number_of_total_frames) - 1

# 동영상 일시 정지 상태 변수
is_paused = False

# ======================================================

# 저장을 위한 비디오 쓰기용 객체 생성
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정 (여기서는 XVID 사용)
out = cv.VideoWriter(SaveFileName, fourcc, fps, (resize_width*2, resize_height))

# =========================================================

# 창 만들기
cv.namedWindow("Video Player : Team 4")

# ======================================================

# 파일 저장 경로 및 파일 이름 지정
save_image_path = Path + 'captured_frames/'
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

# ======================================================

# 트랙바 생성 및 콜백 함수 연결
cv.createTrackbar('Position', 'Video Player : Team 4', 0, max_frame_index, position_callback)

# 초기 밝기 설정
brightness = 10

# 트랙바 생성 및 콜백 함수 연결
cv.createTrackbar('Brightness', 'Video Player : Team 4', brightness, 20, lambda x: None)

# ======================================================

# 창에 마우스 이벤트 콜백 함수 연결
cv.setMouseCallback('Video Player : Team 4', mouse_callback)

# ======================================================

success, frame = videoCapture.read()  # 동영상을 성공적으로 열었을 경우 프레임을 받아온다

current_frame_index = 0
margin = 1  # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

s_time = time.time()  # ms 단위의 현재 tick count을 반환
while success:  # Loop until there are no more frames.
    s = time.time()  # start. time in sec.

    # ======================================================

    # resize 여부에 따른 작업 실행
    if resize_needed:
        resize_frame = cv.resize(frame.copy(), (resize_width, resize_height))
        original = resize_frame.copy()  # 원본 영상
    else:
        original = frame.copy()

    # ======================================================

    # 밝기 트랙바 값 가져오기
    brightness = cv.getTrackbarPos('Brightness', 'Video Player : Team 4')

    # 프레임 밝기 조절
    frame_scaling = cv.convertScaleAbs(original.copy(), alpha=brightness / 10.0)

    # ======================================================

    # AOI 선택된 부분을 표시
    if drawing:
        cv.rectangle(original_copy, roi_start, (mouse_x, mouse_y), (0, 255, 0), 2)

    # ======================================================

    # 현재 프레임 인덱스
    current_frame_index = int(videoCapture.get(cv.CAP_PROP_POS_FRAMES))

    # 현재 프레임 인덱스를 원본 영상과 처리된 영상의 좌측 상단에 빨간색으로 표시
    cv.putText(original, f'org_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv.putText(frame_scaling, f'this_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # ======================================================

    # 화면 분할 기능 : 원본 영상과 scaling된 영상을 x축(가로 방향)상으로 이어붙이기
    new_frame = np.hstack((original, frame_scaling))

    # 분할된 화면 출력하기
    cv.imshow('Video Player : Team 4', new_frame)

    # ======================================================

    # 영상 저장
    out.write(new_frame)  # 현재 프레임 저장

    # ======================================================

    # 재생중인 영상의 프레임 인덱스와 트랙바 위치를 업데이트
    cv.setTrackbarPos('Position', 'Video Player : Team 4', current_frame_index)
    current_frame_index += 1

    # ======================================================

    # 스페이스바 - 정지, esc키 - 종료
    key = cv.waitKey(1)
    if key == 27:  # esc 키를 누르면 비디오 종료
        break
    elif key == ord(' '):  # 스페이스바를 누르면 동영상 일시 정지/재개
        is_paused = not is_paused
        while is_paused:
            key2 = cv.waitKey(1)
            if key2 == ord(' '):  # 스페이스바를 다시 누르면 동영상 일시 정지 해제
                is_paused = False

    # ======================================================
    # 's' 키를 누르면 현재 화면을 이미지로 저장
    if key == ord('s'):
        save_image_name = f'frame_{current_frame_index}.jpg'
        save_image_full_path = os.path.join(save_image_path, save_image_name)
        cv.imwrite(save_image_full_path, new_frame)
        print(f"현재 화면이 {save_image_name}으로 저장되었습니다.")

    # ======================================================

    success, frame = videoCapture.read()  # 다음 프레임을 읽어온다.
    while ((time.time() - s) * 1000) < (dly_ms - margin):  # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

    # ======================================================

    # AOI 처리 기능
    if key == ord('c') and drawing:
        roi = original_copy[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
        roi = cv.bitwise_not(roi)
        original_copy[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = roi
        drawing = False

    elif key == ord('r') and drawing:
        original_copy = original.copy()
        drawing = False

# ======================================================

# 수행시간 출력
e_time = time.time() - s_time
playing_sec = number_of_total_frames / fps  # 상영시간[sec]
print(f'\n\nExpected play time={playing_sec:#.2f}[sec]')
print(f'Real play time={e_time:#.2f}[sec]')

# ======================================================

# 영상 저장 완료 후 릴리즈
videoCapture.release()
out.release()
cv.destroyAllWindows()