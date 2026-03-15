import cv2
import numpy as np
from matplotlib import pyplot as plt

from video_process import get_frame, get_landmarks


def visualize_landmarks(frames, landmarks):
    """
    将landmarks可视化到对应的帧图像上
    """
    visualized_frames = []
    lip_indices = list(range(48, 68))  # 唇部关键点索引范围（按照68个关键点标准，48-67对应唇部）
    orange_color = (0, 165, 255)  # 天蓝色，OpenCV中颜色顺序是BGR
    red_color = (0, 0, 255)  # 红色，OpenCV中颜色顺序是BGR
    green_color = (0, 255, 0)  # 红色，OpenCV中颜色顺序是BGR
    for i in range(len(frames)):
        frame = frames[i].copy()
        landmark = landmarks[i]
        if landmark is not None:
            landmark = landmark[0]  # 取出对应的关键点坐标数据，形状为 (68, 2)
            for j in range(landmark.shape[0]):
                x, y = landmark[j]
                if j in lip_indices:
                    cv2.circle(frame, (int(x), int(y)), 2, green_color, -1)  # 绘制红色的唇部关键点
                else:
                    cv2.circle(frame, (int(x), int(y)), 2, red_color, -1)  # 绘制蓝色的其他关键点
            visualized_frames.append(frame)
    return visualized_frames


def save_images(visualized_images):
    save_path = '/data/LP2/LipReaderClient/app/lipreader/output_images'
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for index, image in enumerate(visualized_images):
        cv2.imwrite(os.path.join(save_path, f'visualized_{index}.png'), image)
    print(f"已将{len(visualized_images)}张可视化图片保存至 {save_path} 路径下。")


if __name__ == "__main__":
    video_path = '/data/LP2/LipReaderClient/app/lipreader/video/section_2_015.58_018.54.mp4'
    frames = get_frame(video_path)
    # 这里假设你有办法获取视频的帧数据，示例中简单构造一个假的帧数据列表（你需要替换成真实读取视频帧的逻辑）
    # video_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    landmarks_result = get_landmarks(frames)
    visualized_images = visualize_landmarks(frames, landmarks_result)
    # 展示可视化后的其中一张图像示例（你可以根据需要选择展示、保存等操作）
    # save
    save_images(visualized_images)
    # show
    plt.imshow(cv2.cvtColor(visualized_images[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
