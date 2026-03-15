import cv2
import face_alignment
import numpy as np
from icecream import ic

from ..utils.util import VideoProcessUtil

import os


# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


def get_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []  # shape: [T, 360, 480, 3]
    if cap.isOpened():
        while True:
            ret, frame = cap.read()  # frame.shape: [360, 480, 3]
            if ret:
                frames.append(frame)
            else:
                break
    else:
        raise IOError("Can't read video:", video_path)

    return frames


def get_landmarks(frames):
    face = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda"
    )
    # 人脸关键点检测
    landmarks = []  # shape: [T, 2, 68]
    for frame in frames:
        points_list = face.get_landmarks(frame)
        landmarks.append(points_list)

    return landmarks


def get_lips(frames, landmarks):
    # 提取嘴唇
    lips = []
    front256 = VideoProcessUtil.get_position(256)
    for i, (frame, landmark) in enumerate(zip(frames, landmarks)):
        anno = sorted(landmark, key=VideoProcessUtil.cal_area, reverse=True)[0]
        shape = anno[17:]

        M = VideoProcessUtil.transformation_from_points(np.matrix(shape), np.matrix(front256))
        img = cv2.warpAffine(frame, M[:2], (256, 256))
        (x, y) = front256[-20:].mean(0).astype(np.int32)
        w = 160 // 2
        img = img[y - w // 2 : y + w // 2, x - w : x + w, ...]  # shape: [80, 160, 3]
        lips.append(img)
        if i == 0:
            ic(img.shape)
        # cv2.imwrite(os.path.join(path_target, '{}.jpg'.format(i)), img)
    return lips


def read_lips(video_path):
    frames = get_frame(video_path)
    print("get frames.")
    landmarks = get_landmarks(frames)
    print("get landmarks.")
    lips = get_lips(frames, landmarks)
    print("get lips.")

    return np.array(lips)


if __name__ == "__main__":
    video_path = "video/section_1_000.80_002.91.mp4"
    read_lips(video_path)
