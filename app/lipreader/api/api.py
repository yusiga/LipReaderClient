from icecream import ic
import time
from ..utils import VideoProcessUtil, ModelInferenceUtil


def lipreading_inference(video_file_list, device, language):
    # * 1 video process
    time1 = time.time()
    frame_list = [VideoProcessUtil.get_frame(file) for file in video_file_list]
    time2 = time.time()
    ic("get frame time: ", time2 - time1)
    ic("get frame list")
    landmark_list = [VideoProcessUtil.get_landmarks(frame, device) for frame in frame_list]
    time3 = time.time()
    ic("get landmarks time: ", time3 - time2)
    ic("get landmark list")
    lip_list = [
        VideoProcessUtil.get_lips(frame, landmark)
        for frame, landmark in zip(frame_list, landmark_list)
    ]
    time4 = time.time()
    ic("get lip time: ", time4 - time3)
    ic("get lip list")

    # * 2 make video mask
    video_lip, video_mask = ModelInferenceUtil.prepare_data(lip_list)  # (B, C, T, H, W) | (B, T)
    time5 = time.time()
    ic("get video mask time: ", time5 - time4)

    # * 3 inference
    predict_text_list = ModelInferenceUtil.lipreading_inference(
        video_lip, video_mask, device, language
    )
    time6 = time.time()
    ic("get text time: ", time6 - time5)

    return predict_text_list


def kws_inference(video_file_list, device, strategy, top_k, clip):
    # * 1 video process
    frame_list = [VideoProcessUtil.get_frame(file) for file in video_file_list]
    ic("get frame list")
    landmark_list = [VideoProcessUtil.get_landmarks(frame, device) for frame in frame_list]
    ic("get landmark list")
    lip_list = [
        VideoProcessUtil.get_lips(frame, landmark)
        for frame, landmark in zip(frame_list, landmark_list)
    ]
    ic("get lip list")

    # * 2 mask video mask
    video_lip, video_mask = ModelInferenceUtil.prepare_data(lip_list)  # (B, C, T, H, W) | (B, T)

    # * 3 kws inference (B, clip_group, top_k)
    predict_text_list = ModelInferenceUtil.kws_inference(
        video_lip, video_mask, device, strategy, clip, top_k
    )
    return predict_text_list
