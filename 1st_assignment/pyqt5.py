import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QSlider, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QTime

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")

        self.video_capture = cv2.VideoCapture()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.playing = False  # 영상 재생 여부를 나타내는 플래그

        self.play_button = QPushButton("Pause Video")
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.hide()

        self.Open_button = QPushButton("Open Video")
        self.Open_button.clicked.connect(self.open_video)

        # 슬라이더 위젯 생성
        self.slider1 = QSlider()
        self.slider1.setMinimum(0)  # 최소값 설정
        self.slider1.setMaximum(20)  # 최대값 설정
        self.slider1.setValue(10)  # 초기값 설정
        self.slider1.setOrientation(1)  # 수평 방향으로 설정 (0은 수직)
        self.slider1.hide()

        # 슬라이더 위젯 생성
        self.slider2 = QSlider()
        self.slider2.setMinimum(0)  # 최소값 설정
        self.slider2.setMaximum(100)  # 최대값 설정
        self.slider2.setValue(0)  # 초기값 설정
        self.slider2.setOrientation(1)  # 수평 방향으로 설정 (0은 수직)
        self.slider2.hide()

        # 슬라이더 값 변경 시 이벤트 핸들러 연결
        self.slider2.valueChanged.connect(self.slider2_value_changed)

        self.save_button = QPushButton("Save Video")  # 추가: 영상 저장 버튼
        self.save_button.clicked.connect(self.save_video)
        self.save_button.hide()

        self.frame_label = QLabel(self)
        self.current_time_label = QLabel(self)  # 현재 재생 시간을 표시할 레이블 위젯 생성
        self.brightness = QLabel(self)  # 현재 밝기

        self.h_layout1 = QHBoxLayout()
        self.h_layout2 = QHBoxLayout()

        # 수평 레이아웃에 위젯 추가
        self.h_layout1.addWidget(self.brightness)
        self.h_layout1.addWidget(self.slider1)

        # 수평 레이아웃에 위젯 추가
        self.h_layout2.addWidget(self.current_time_label)
        self.h_layout2.addWidget(self.slider2)

        # 수직 레이아웃 생성
        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.play_button)
        self.v_layout.addWidget(self.Open_button)
        self.v_layout.addWidget(self.frame_label)
        self.v_layout.addLayout(self.h_layout1)  # 수평 레이아웃 추가
        self.v_layout.addLayout(self.h_layout2)  # 수평 레이아웃 추가
        self.v_layout.addWidget(self.save_button)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.v_layout)
        self.setCentralWidget(self.central_widget)

        self.frames = []  # 추가: 프레임을 저장할 리스트

        self.current_time = QTime(0, 0)  # 현재 시간 초기화

    def open_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if video_file:
            self.video_capture.open(video_file)
            if self.video_capture.isOpened():
                self.play_button.show()
                self.slider1.show()
                self.slider2.show()
                self.save_button.show()
                self.timer.start(30)  # Update frame every 30 milliseconds
                self.playing = True
                self.frames = []  # 비디오를 다시 열 때 프레임 리스트 초기화
                self.Open_button.hide()  # Play 버튼 숨기기

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText("Pause Video")
            self.timer.start(30)  # 영상 재생 시작
        else:
            self.play_button.setText("Play Video")
            self.timer.stop()  # 영상 재생 일시정지

    def update_frame(self):
        ret, frame = self.video_capture.read()
        self.slider2.setMaximum(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.set_slider_position(int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if ret:
            original = frame.copy()  # frame1에 원본 프레임 복사
            frame_scaling = cv2.convertScaleAbs(frame.copy(), alpha= self.slider1.value() / 10.0)

            cv2.putText(original, f'org_index={int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))}', (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame_scaling, f'this_index={int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))}', (10, 15), cv2.FONT_HERSHEY_PLAIN, 1,
                       (0, 0, 255), 2)

            self.brightness.setText(f'brightness : {self.slider1.value()}')

            # 두 개의 프레임을 가로로 합치기
            combined_frame = np.hstack((original, frame_scaling ))

            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = combined_frame.shape
            bytes_per_line = 3 * width
            qt_image = QImage(combined_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_label.setPixmap(pixmap)

            combined_frame1 = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            self.frames.append(combined_frame1)  # 현재 프레임을 리스트에 추가

            # 현재 프레임 번호와 프레임 레이트 가져오기
            current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))

            # 현재 재생 시간 계산
            current_time_seconds = current_frame / frame_rate

            # 현재 재생 시간을 QTime으로 업데이트
            self.current_time = QTime(0, 0)
            self.current_time = self.current_time.addSecs(int(current_time_seconds))

            # 현재 시간을 문자열로 변환하여 출력
            time_str = self.current_time.toString("hh:mm:ss")

            self.current_time_label.setText(time_str)

    def save_video(self):
        if len(self.frames) > 0:
            height, width, channel = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('4조.avi', fourcc, 30.0, (width, height))

            for frame in self.frames:
                out.write(frame)

            out.release()
            print("Video saved as combined_video.avi")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:  # 스페이스바를 누르면 재생/일시정지 토글
            self.toggle_play()
        elif event.key() == Qt.Key_Escape:  # ESC 키를 누르면 창을 닫음
            self.close()

    def slider2_value_changed(self):
        # 슬라이더 값 변경 시 호출되는 함수
        value = self.slider2.value()
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, value)

    def set_slider_position(self, value):
        # 슬라이더의 현재 값을 변경하는 함수
        self.slider2.setSliderPosition(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
