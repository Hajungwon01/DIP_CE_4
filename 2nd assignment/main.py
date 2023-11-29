"""
제출자 성명: 하정원, 정성원, 박해윤
제출자 학과 및 학번: 컴퓨터공학과 2020305082, 컴퓨터공학과 2020305068, 컴퓨터공학과 2018305028

※ 구현 목록 ※
    (1) playing ✔️
    (2) forwarding, backwarding - Trackbar로 제어 ✔️
    (3) 정지 - 스페이스바 ✔️
    (4) 종료 - esc키 ✔️
    (5) 재생 중 인덱스 번호 출력 기능 ✔️
    (6) 동영상 저장 기능 ✔️
    (7) AOI => 7-1 ✔️7-2 ✔️
    (8) 영상처리 알고리즘(기본) - HE ✔️
    (9) 영상처리 알고리즘(추가) - 3가지 중 하나 => BF ✔️UM ✔️

"""

# 변수 선언부
Path = '../data/'
Name = 'matrix.avi'

# ======================================================
#필요한 모듈 import

import cv2 as cv
import time
import numpy as np
import os

# ======================================================
# 트랙바 콜백 함수
def position_callback(pos):
    videoCapture.set(cv.CAP_PROP_POS_FRAMES, pos)

def callback_AlgSelect(x):
    pass

def callback_sigma(x):
    pass

def callback_scale(x):
    pass

def callback_sigmaColor(x):
    pass

# ======================================================
# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global s_x, s_y, e_x, e_y, drawing_needed

    if event == cv.EVENT_LBUTTONDOWN: # 마우스의 왼쪽 버튼이 눌렸을 때
        s_x, s_y = x, y # ROI의 시작점
        drawing_needed = False # 아직 ROI 선택이 완료되지 않음

    elif event == cv.EVENT_LBUTTONUP: # 마우스의 왼쪽 버튼이 떼어졌을 때
        e_x, e_y = x, y # ROI의 끝점
        drawing_needed = True # ROI 선택 완료

# ======================================================
# 필터링 함수
# (1) Histogram Equalization
def HE(s_x, s_y, e_x, e_y, img):
    global drawing_needed

    if drawing_needed: # ROI 선택되었을 때
        img1 = img.copy()
        # 각각의 BGR 채널에 대해 HE를 실행
        equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img1[s_y:e_y, s_x:e_x])]
        # HE를 실행한 채널들을 결합
        equalized_img = cv.merge(equalized_channels)
        img1[s_y:e_y, s_x:e_x] = equalized_img
        return img1
    else: # ROI가 선택되지 않았을 때
        # 각각의 BGR 채널에 대해 HE를 실행
        equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img)]
        # HE를 실행한 채널들을 결합
        equalized_img = cv.merge(equalized_channels)
        return equalized_img

# (2) Bilateral Filtering
def BF(s_x, s_y, e_x, e_y, img):
    global drawing_needed
    d = -1
    sigmaColor = cv.getTrackbarPos('SigmaColor',
                                   'Video Editing Program : Team 4')
    sigmaSpace = 13
    if drawing_needed: # ROI 선택되었을 때
        img1 = img.copy()
        dst = cv.bilateralFilter(img1[s_y:e_y, s_x:e_x], d, sigmaColor, sigmaSpace)
        img1[s_y:e_y, s_x:e_x] = dst
        return img1
    else: # ROI가 선택되지 않았을 때
        dst = cv.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        return dst

# (3) Unsharp Masking
def UM(s_x, s_y, e_x, e_y, img):
    global drawing_needed
    if drawing_needed: # ROI 선택되었을 때
        img1 = img.copy()
        k = cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4') * 6 + 1
        blur = cv.GaussianBlur(src=img1[s_y:e_y, s_x:e_x], ksize=(k, k), sigmaX=cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4'))
        UnsharpMaskImg = img1[s_y:e_y, s_x:e_x] - blur
        SharpenedImg = img1[s_y:e_y, s_x:e_x] + cv.getTrackbarPos('Scale', 'Video Editing Program : Team 4') * UnsharpMaskImg
        img1[s_y:e_y, s_x:e_x] = SharpenedImg
        return img1
    else: # ROI가 선택되지 않았을 때
        k = cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4') * 6 + 1
        blur = cv.GaussianBlur(src=img, ksize=(k, k), sigmaX=cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4'))
        UnsharpMaskImg = img - blur
        SharpenedImg = img + cv.getTrackbarPos('Scale', 'Video Editing Program : Team 4') * UnsharpMaskImg
        return SharpenedImg

# ======================================================

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
dly_ms = 1000 / (fps)

# 동영상의 너비와 높이
width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))

# 최대 프레임 인덱스
max_frame_index = 1000

# 동영상 일시 정지 상태 변수
is_paused = False

# ROI 시작점, 끝점 초기화
s_x = s_y = e_x = e_y = -1

# ROI 선택 여부
drawing_needed = False

# 적용 가능한 알고리즘 리스트
algorithm = ["Default", "Histogram Equalization", "Bilateral Filtering", "Unsharp Masking"]

# ======================================================
# 저장을 위한 비디오 쓰기용 객체 생성
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정 (여기서는 XVID 사용)
out = cv.VideoWriter(SaveFileName, fourcc, fps, (width * 2, height))

# =========================================================
# 창 만들기
cv.namedWindow("Video Editing Program : Team 4")
cv.setMouseCallback("Video Editing Program : Team 4", mouse_callback)

# ======================================================
# 현재 화면(정지영상) 저장
save_image_path = Path + 'captured_frames/'
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

# ======================================================
# 트랙바 생성 및 콜백 함수 연결
# position
cv.createTrackbar('Position', 'Video Editing Program : Team 4', 0, max_frame_index, position_callback)
# algorithm
cv.createTrackbar('Algorithm', 'Video Editing Program : Team 4', 0, 3, callback_AlgSelect)
# UM - sigma
cv.createTrackbar('Sigma', 'Video Editing Program : Team 4', 1, 15, callback_sigma)
# UM - scale
cv.createTrackbar('Scale', 'Video Editing Program : Team 4', 1, 6, callback_scale)
# BF - sigmaColor
cv.createTrackbar('SigmaColor', 'Video Editing Program : Team 4', 25, 50, callback_sigmaColor)

# ======================================================
# 영상 재생
success, frame = videoCapture.read()  # 동영상을 성공적으로 열었을 경우 프레임을 받아온다

count = 0 # 우측 영상의 인덱스
current_frame_index = 0 # 좌측 영상의 인덱스
margin = 1  # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

while success:  # Loop until there are no more frames.
    s = time.time()  # start. time in sec.

    frame_scaling = frame.copy()

    count += 1

    # ======================================================

    if drawing_needed: # ROI 선택했다면
        if s_y > e_y:  # y축상의 시작점과 끝점이 바뀌었으면 두 좌표를 바꾼다.
            s_y, e_y = e_y, s_y
        if s_x > e_x:  # x축상의 시작점과 끝점이 바뀌었으면 두 좌표를 바꾼다.
            s_x, e_x = e_x, s_x
        cv.rectangle(frame_scaling, (s_x, s_y), (e_x, e_y), (0, 255, 0), 1) # ROI를 사각형으로 그린다.

    # ======================================================
    # 선택된 알고리즘에 따른 작업 수행
    x = cv.getTrackbarPos('Algorithm', 'Video Editing Program : Team 4')
    chosen = algorithm[x]

    if x == 0:      # default
        pass
    elif x == 1:    # HE
        frame_scaling = HE(s_x, s_y, e_x, e_y, frame_scaling)
    elif x == 2:    # BF
        frame_scaling = BF(s_x, s_y, e_x, e_y, frame_scaling)
    elif x == 3:    # UM
        frame_scaling = UM(s_x, s_y, e_x, e_y, frame_scaling / 255)
        frame_scaling = np.clip(frame_scaling * 255, 0, 255).astype('uint8')

    # ======================================================
    # 현재 프레임 인덱스
    current_frame_index = int(videoCapture.get(cv.CAP_PROP_POS_FRAMES))

    # 현재 프레임 인덱스를 원본 영상과 처리된 영상의 좌측 상단에 빨간색으로 표시
    cv.putText(frame, f'org_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv.putText(frame_scaling, f'this_index={count}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
               2)

    # 현재 선택된 알고리즘 처리된 영상의 우측 하단에 보라색으로 표시
    cv.putText(frame_scaling, f'ALGORITHM : {chosen}',
               (frame_scaling.shape[1] - 220 - len(chosen), frame_scaling.shape[0] - 19),
               cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255),
               2)

    # ROI 선택되었다면 ROI 좌표도 함께 처리된 영상의 우측 하단에 보라색으로 표시
    if drawing_needed:
        cv.putText(frame_scaling, f'coordinate : ({s_x}, {s_y}), ({e_x}, {e_y})', (frame_scaling.shape[1] - 220 - len(chosen), frame_scaling.shape[0] - 5),
                   cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255),
                   2)
    # ======================================================
    # 화면 분할 기능 : 원본 영상과 scaling된 영상을 x축(가로 방향)상으로 이어붙이기
    new_frame = np.hstack((frame, frame_scaling))

    # 분할된 화면 출력하기
    cv.imshow('Video Editing Program : Team 4', new_frame)

    # ======================================================
    # 영상 저장
    out.write(new_frame)  # 현재 프레임 저장

    # ======================================================
    # 재생중인 영상의 프레임 인덱스와 트랙바 위치를 업데이트
    cv.setTrackbarPos('Position', 'Video Editing Program : Team 4', current_frame_index)

    # ======================================================
    # 스페이스바 - 정지, esc키 - 종료
    key = cv.waitKey(1)

    if key == 27:  # esc 키를 누르면 비디오 종료
        break
    elif key == ord(' '):  # 스페이스바를 누르면 동영상 일시 정지/재개
        is_paused = not is_paused
        while is_paused:
            key2 = cv.waitKey(1)
            if key2 != -1:  # 아무 키나 누르면 동영상 일시 정지 해제
                is_paused = False
    elif key == ord('s'):   # 's' 키를 누르면 현재 화면을 이미지로 저장
        save_image_name = f'frame_{current_frame_index}.jpg'
        save_image_full_path = os.path.join(save_image_path, save_image_name)
        cv.imwrite(save_image_full_path, new_frame)
        print(f"현재 화면이 {save_image_name}으로 저장되었습니다.")

    # ======================================================

    current_frame_index += 1
    success, frame = videoCapture.read()  # 다음 프레임을 읽어온다.

    # ======================================================
    while ((time.time() - s) * 1000) < (dly_ms - margin):  # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

    # ======================================================

videoCapture.release()
out.release()
cv.destroyAllWindows()

# (7-1) 사용 구현 코드
"""
# 변수 선언부
Path = '../data/'
Name = 'matrix.avi'

# ======================================================
#필요한 모듈 import

import cv2 as cv
import time
import numpy as np
import os

# ======================================================
# 트랙바 콜백 함수
def position_callback(pos):
    videoCapture.set(cv.CAP_PROP_POS_FRAMES, pos)

def callback_AlgSelect(x):
    pass

def callback_sigma(x):
    pass

def callback_scale(x):
    pass

def callback_sigmaColor(x):
    pass

# ======================================================
# 필터링 함수
# (1) Histogram Equalization
def HE(roi, img):
    if roi is not None:
        img1 = img.copy()
        # 각각의 BGR 채널에 대해 HE를 실행
        equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])])]

        # HE를 실행한 채널들을 결합
        equalized_img = cv.merge(equalized_channels)

        img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = equalized_img
        return img1
    else:
        # 각각의 BGR 채널에 대해 HE를 실행
        equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img)]

        # HE를 실행한 채널들을 결합
        equalized_img = cv.merge(equalized_channels)
        return equalized_img

# (2) Bilateral Filtering
def BF(roi, img):
    d = -1
    sigmaColor = cv.getTrackbarPos('SigmaColor',  # 트랙바 앞에 표시될 트랙바의 이름
                                   'Video Editing Program : Team 4')
    sigmaSpace = 7
    if roi is not None:
        img1 = img.copy()
        dst = cv.bilateralFilter(img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], d, sigmaColor, sigmaSpace)
        img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = dst
        return img1
    else:
        dst = cv.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        return dst

# (3) Unsharp Masking
def UM(roi, img):
    if roi is not None:
        img1 = img.copy()
        k = cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4') * 6 + 1
        blur = cv.GaussianBlur(src=img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], ksize=(k, k), sigmaX=cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4'))
        UnsharpMaskImg = img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] - blur
        SharpenedImg = img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] + cv.getTrackbarPos('Scale', 'Video Editing Program : Team 4') * UnsharpMaskImg
        img1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = SharpenedImg
        return img1
    else:
        k = cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4') * 6 + 1
        blur = cv.GaussianBlur(src=img, ksize=(k, k), sigmaX=cv.getTrackbarPos('Sigma', 'Video Editing Program : Team 4'))
        UnsharpMaskImg = img - blur
        SharpenedImg = img + cv.getTrackbarPos('Scale', 'Video Editing Program : Team 4') * UnsharpMaskImg
        return SharpenedImg

# ======================================================

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
dly_ms = 1000 / (fps)

# 동영상의 너비와 높이
width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))

# 최대 프레임 인덱스
max_frame_index = 1000

# 동영상 일시 정지 상태 변수
is_paused = False

# 적용 가능한 알고리즘 리스트
algorithm = ["Default", "Histogram Equalization", "Bilateral Filtering", "Unsharp Masking"]

# ======================================================
# 저장을 위한 비디오 쓰기용 객체 생성
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정 (여기서는 XVID 사용)
out = cv.VideoWriter(SaveFileName, fourcc, fps, (width * 2, height))

# =========================================================
# 창 만들기
cv.namedWindow("Video Editing Program : Team 4")

# ======================================================
# 파일 저장 경로 및 파일 이름 지정
save_image_path = Path + 'captured_frames/'
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

# ======================================================
# 트랙바 생성 및 콜백 함수 연결
# position
cv.createTrackbar('Position', 'Video Editing Program : Team 4', 0, max_frame_index, position_callback)
# algorithm
cv.createTrackbar('Algorithm', 'Video Editing Program : Team 4', 0, 3, callback_AlgSelect)
# UM - sigma
cv.createTrackbar('Sigma', 'Video Editing Program : Team 4', 1, 15, callback_sigma)
# UM - scale
cv.createTrackbar('Scale', 'Video Editing Program : Team 4', 1, 6, callback_scale)
# BF - sigmaColor
cv.createTrackbar('SigmaColor', 'Video Editing Program : Team 4', 25, 50, callback_sigmaColor)

# ======================================================
# AOI 선택을 위한 전역 변수
roi = None
original_copy = None

# ======================================================
# 영상 재생
success, frame = videoCapture.read()  # 동영상을 성공적으로 열었을 경우 프레임을 받아온다

count = 0 # 우측 영상의 인덱스
current_frame_index = 0 # 좌측 영상의 인덱스
margin = 1  # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

while success:  # Loop until there are no more frames.
    s = time.time()  # start. time in sec.

    frame_scaling = frame.copy()

    count += 1

    # ======================================================
    # 선택된 알고리즘에 따른 작업 수행
    x = cv.getTrackbarPos('Algorithm', 'Video Editing Program : Team 4')
    chosen = algorithm[x]

    if x == 0:  # default
        pass
    elif x == 1:  # HE
        frame_scaling = HE(roi, frame_scaling)
    elif x == 2:  # BF
        frame_scaling = BF(roi, frame_scaling)
    elif x == 3:  # UM
        frame_scaling = UM(roi, frame_scaling / 255)
        frame_scaling = np.clip(frame_scaling * 255, 0, 255).astype('uint8')

    # ======================================================
    # 현재 프레임 인덱스
    current_frame_index = int(videoCapture.get(cv.CAP_PROP_POS_FRAMES))

    # 현재 프레임 인덱스를 원본 영상과 처리된 영상의 좌측 상단에 빨간색으로 표시
    cv.putText(frame, f'org_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv.putText(frame_scaling, f'this_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
               2)

    # 현재 선택된 알고리즘 처리된 영상의 우측 하단에 보라색으로 표시
    cv.putText(frame_scaling, f'ALGORITHM : {chosen}',
               (frame_scaling.shape[1] - 220 - len(chosen), frame_scaling.shape[0] - 19),
               cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255),
               2)

    # ======================================================
    # ROI에 사각형 만들기
    if roi is not None: # ROI 선택되었다면 ROI 좌표도 함께 처리된 영상의 우측 하단에 보라색으로 표시
        cv.rectangle(frame_scaling, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0),2)
        cv.putText(frame_scaling, f'coordinate : ({int(roi[0])}, {int(roi[1])}), ({int(roi[0] + roi[2])}, {int(roi[1] + roi[3])})',
                   (frame_scaling.shape[1] - 220 - len(chosen), frame_scaling.shape[0] - 5),
                   cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255),
                   2)

    # ======================================================
    # 화면 분할 기능 : 원본 영상과 scaling된 영상을 x축(가로 방향)상으로 이어붙이기
    new_frame = np.hstack((frame, frame_scaling))

    # 분할된 화면 출력하기
    cv.imshow('Video Editing Program : Team 4', new_frame)

    # ======================================================
    # 영상 저장
    out.write(new_frame)  # 현재 프레임 저장

    # ======================================================
    # 재생중인 영상의 프레임 인덱스와 트랙바 위치를 업데이트
    cv.setTrackbarPos('Position', 'Video Editing Program : Team 4', current_frame_index)

    # ======================================================

    current_frame_index += 1
    success, frame = videoCapture.read()  # 다음 프레임을 읽어온다.

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
    elif key == ord('s'): # 's' 키를 누르면 현재 화면을 이미지로 저장
        save_image_name = f'frame_{current_frame_index}.jpg'
        save_image_full_path = os.path.join(save_image_path, save_image_name)
        cv.imwrite(save_image_full_path, new_frame)
        print(f"현재 화면이 {save_image_name}으로 저장되었습니다.")
    elif key == ord('r'): # 'r' 키를 누르면 ROI 선택 모드로 전환
        print("ROI 선택 모드로 전환. 화면에서 드래그하여 ROI를 선택하세요.")
        roi = cv.selectROI("Video Editing Program : Team 4", frame_scaling, showCrosshair=False)

    # ======================================================
    while ((time.time() - s) * 1000) < (dly_ms - margin):  # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

    # ======================================================

videoCapture.release()
out.release()
cv.destroyAllWindows()
"""
