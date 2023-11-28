import cv2 as cv
import time
import numpy as np
import os

# AOI 선택을 위한 전역 변수
roi = None
drawing = False
roi_start = (0, 0)
roi_end = (0, 0)
original_copy = None

# position 트랙바 콜백 함수
def position_callback(pos):
    videoCapture.set(cv.CAP_PROP_POS_FRAMES, pos)

# 시그마 트랙바 콜백 함수
def callback_S(x):
    pass

# 스케일 트랙바 콜백 함수
def callback_S1(x):
    pass

# 시그마 컬러 트랙바 콜백 함수
def BF_sigmaColor(x):
    pass

def UM(img):
    k = cv.getTrackbarPos('sigma', 'Video Player : Team 4') * 6 + 1
    blur = cv.GaussianBlur(src=img, ksize=(k, k), sigmaX=cv.getTrackbarPos('sigma', 'Video Player : Team 4'))
    UnsharpMaskImg = img - blur
    SharpenedImg = img + cv.getTrackbarPos('scale', 'Video Player : Team 4') * UnsharpMaskImg
    return SharpenedImg

def BF(img):
    d = cv.getTrackbarPos('sigma', 'Video Player : Team 4') * 6 + 1
    sigmaColor = cv.getTrackbarPos('sigmaColor', 'Video Player : Team 4')
    sigmaSpace = cv.getTrackbarPos('sigma', 'Video Player : Team 4')
    dst = cv.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    return dst

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
dly_ms = 1000 / (fps)

# 동영상의 너비와 높이
width = int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))

# 최대 프레임 인덱스
max_frame_index = int(number_of_total_frames) - 1

# 동영상 일시 정지 상태 변수
is_paused = False

# ======================================================

# 저장을 위한 비디오 쓰기용 객체 생성
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정 (여기서는 XVID 사용)
out = cv.VideoWriter(SaveFileName, fourcc, fps, (width * 2, height))

# =========================================================

# 창 만들기
cv.namedWindow("Video Player : Team 4")

# ======================================================

# 파일 저장 경로 및 파일 이름 지정
save_image_path = Path + 'captured_frames/'
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

# 트랙바 생성 및 콜백 함수 연결
cv.createTrackbar('Position', 'Video Player : Team 4', 0, max_frame_index, position_callback)
cv.createTrackbar('sigma', 'Video Player : Team 4', 1, 8, callback_S)
cv.createTrackbar('scale', 'Video Player : Team 4', 1, 6, callback_S1)
cv.createTrackbar('sigmaColor', 'Video Player : Team 4', 1, 15, BF_sigmaColor)

success, frame = videoCapture.read()  # 동영상을 성공적으로 열었을 경우 프레임을 받아온다

current_frame_index = 0
margin = 1  # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

s_time = time.time()  # ms 단위의 현재 tick count을 반환
while success:  # Loop until there are no more frames.
    s = time.time()  # start. time in sec.

    # ======================================================

    frame_scaling = frame.copy()

    # ======================================================

    # 현재 프레임 인덱스
    current_frame_index = int(videoCapture.get(cv.CAP_PROP_POS_FRAMES))

    # frame_scaling = UM(frame_scaling / 255)
    frame_scaling = BF(frame_scaling)
    # 현재 프레임 인덱스를 원본 영상과 처리된 영상의 좌측 상단에 빨간색으로 표시
    cv.putText(frame, f'org_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv.putText(frame_scaling, f'this_index={current_frame_index}', (10, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
               2)

    # ======================================================

    # 화면 분할 기능 : 원본 영상과 scaling된 영상을 x축(가로 방향)상으로 이어붙이기
    # frame_scaling = np.clip(frame_scaling * 255, 0, 255).astype('uint8')
    new_frame = np.hstack((frame, frame_scaling))

    # 분할된 화면 출력하기
    cv.imshow('Video Player : Team 4', new_frame)

    # ======================================================

    # 영상 저장
    out.write(new_frame)  # 현재 프레임 저장

    # ======================================================

    # AOI 선택
    if roi is not None:
        roi_frame = frame_scaling[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv.imshow('ROI', roi_frame)

    # ======================================================

    # 재생중인 영상의 프레임 인덱스와 트랙바 위치를 업데이트
    cv.setTrackbarPos('Position', 'Video Player : Team 4', current_frame_index)
    current_frame_index += 1

    # ======================================================
    success, frame = videoCapture.read()  # 다음 프레임을 읽어온다.

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
    elif key == ord('s'):
        save_image_name = f'frame_{current_frame_index}.jpg'
        save_image_full_path = os.path.join(save_image_path, save_image_name)
        cv.imwrite(save_image_full_path, new_frame)
        print(f"현재 화면이 {save_image_name}으로 저장되었습니다.")
    # 'r' 키를 누르면 ROI 선택 모드로 전환
    elif key == ord('r'):
        print("ROI 선택 모드로 전환. 화면에서 드래그하여 ROI를 선택한 후 q키를 누르세요.")
        original_copy = frame.copy()  # 원본 프레임 복사
        roi_start = (0, 0)
        roi_end = (0, 0)
        drawing = True

    elif key == ord('q') and drawing:  # q를 누르면 ROI 선택 완료
        roi_end = (int(roi_end[0] + roi_start[0]), int(roi_end[1] + roi_start[1]))
        roi = tuple(map(int, (roi_start[0], roi_start[1], roi_end[0] - roi_start[0], roi_end[1] - roi_start[1])))
        drawing = False
        cv.rectangle(original_copy, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        cv.imshow("Video Player : Team 4", original_copy)

    # ======================================================
    while ((time.time() - s) * 1000) < (dly_ms - margin):  # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

# 동영상 재생이 끝나면 종료
videoCapture.release()
out.release()
cv.destroyAllWindows()
