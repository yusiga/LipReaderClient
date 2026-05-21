from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
import face_alignment
import os
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d
from icecream import ic

# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from ..config import args
from ..assets import hanzi_list, pinyin_list
from ..model import C3dModel, WPELipReader

# from .hanzi import hanzi_list
# from .pinyin import pinyin_list
# from .model.model_c3d import C3dModel
# from .model.model_wpelip import LipReader as ManuKws
# from .model.model_autoKws import LipReader as AutoKws


# from hanzi import hanzi_list
# from pinyin import pinyin_list
# from model.model_c3d import C3dModel
# from model.model_wpelip import  ManuKws


class VideoProcessUtil:
    @staticmethod
    def read_lips(video_path):
        frames = VideoProcessUtil.get_frame(video_path)
        print("get frames.")
        landmarks = VideoProcessUtil.get_landmarks(frames)
        print("get landmarks.")
        lips = VideoProcessUtil.get_lips(frames, landmarks)
        print("get lips.")
        return np.array(lips)  # (T, H, W, C)

    @staticmethod
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

    @staticmethod
    def get_landmarks(frames, device):
        face = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )
        frames = np.stack(frames)
        frames = torch.from_numpy(frames)
        frames = frames.to(device)
        landmarks = [face.get_landmarks(frame) for frame in frames]

        # 使用线程池并行处理
        # with ThreadPoolExecutor() as executor:
        #     landmarks = list(executor.map(face.get_landmarks, frames))

        return landmarks

    @staticmethod
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
            # if i == 0:
            #     ic(img.shape)
            # cv2.imwrite(os.path.join(path_target, '{}.jpg'.format(i)), img)
        return lips

    @staticmethod
    def transformation_from_points(points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack(
            [np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0.0, 0.0, 1.0])]
        )

    @staticmethod
    def get_position(size, padding=0.25):
        x = [
            0.000213256,
            0.0752622,
            0.18113,
            0.29077,
            0.393397,
            0.586856,
            0.689483,
            0.799124,
            0.904991,
            0.98004,
            0.490127,
            0.490127,
            0.490127,
            0.490127,
            0.36688,
            0.426036,
            0.490127,
            0.554217,
            0.613373,
            0.121737,
            0.187122,
            0.265825,
            0.334606,
            0.260918,
            0.182743,
            0.645647,
            0.714428,
            0.793132,
            0.858516,
            0.79751,
            0.719335,
            0.254149,
            0.340985,
            0.428858,
            0.490127,
            0.551395,
            0.639268,
            0.726104,
            0.642159,
            0.556721,
            0.490127,
            0.423532,
            0.338094,
            0.290379,
            0.428096,
            0.490127,
            0.552157,
            0.689874,
            0.553364,
            0.490127,
            0.42689,
        ]

        y = [
            0.106454,
            0.038915,
            0.0187482,
            0.0344891,
            0.0773906,
            0.0773906,
            0.0344891,
            0.0187482,
            0.038915,
            0.106454,
            0.203352,
            0.307009,
            0.409805,
            0.515625,
            0.587326,
            0.609345,
            0.628106,
            0.609345,
            0.587326,
            0.216423,
            0.178758,
            0.179852,
            0.231733,
            0.245099,
            0.244077,
            0.231733,
            0.179852,
            0.178758,
            0.216423,
            0.244077,
            0.245099,
            0.780233,
            0.745405,
            0.727388,
            0.742578,
            0.727388,
            0.745405,
            0.780233,
            0.864805,
            0.902192,
            0.909281,
            0.902192,
            0.864805,
            0.784792,
            0.778746,
            0.785343,
            0.778746,
            0.784792,
            0.824182,
            0.831803,
            0.824182,
        ]

        x, y = np.array(x), np.array(y)

        x = (x + padding) / (2 * padding + 1)
        y = (y + padding) / (2 * padding + 1)
        x = x * size
        y = y * size
        return np.array(list(zip(x, y)))

    @staticmethod
    def cal_area(anno):
        return (anno[:, 0].max() - anno[:, 0].min()) * (anno[:, 1].max() - anno[:, 1].min())


class ModelInferenceUtil:
    pad_token, sos_token, eos_token, blank_token = 0, 1, 2, 3
    hanzi_vocab = ["<pad>", "<sos>", "<eos>", "-"] + hanzi_list
    pinyin_vocab = ["<pad>", "<sos>", "<eos>", "-"] + pinyin_list

    @staticmethod
    def smooth_prediction_segments(pred, min_segment_len=2):
        """
        平滑Top1序列，例如：451 451 1754 451 451 → 451 451 451 451 451
        """

        pred = pred.clone()
        T = len(pred)
        start = 0
        segments = []

        for i in range(1, T):
            if pred[i] != pred[i - 1]:
                segments.append([start, i, pred[i - 1].item()])
                start = i

        segments.append([start, T, pred[-1].item()])

        for idx in range(1, len(segments) - 1):
            seg_start, seg_end, seg_cls = segments[idx]
            seg_len = seg_end - seg_start

            prev_cls = segments[idx - 1][2]
            next_cls = segments[idx + 1][2]

            if seg_len < min_segment_len:
                if prev_cls == next_cls:
                    pred[seg_start:seg_end] = prev_cls

        return pred

    @staticmethod
    def extract_change_boundaries(pred):
        """
        从Top1类别变化中提取边界
        """

        boundaries = [0]
        for t in range(1, len(pred)):
            if pred[t] != pred[t - 1]:
                if pred[t] == ModelInferenceUtil.blank_token:
                    continue
                boundaries.append(t)
        boundaries.append(len(pred))
        return boundaries

    @staticmethod
    def refine_boundaries_with_js(boundaries, js_curve, search_radius=2):
        """
        将边界对齐到附近JS峰值
        """

        js_curve = (
            js_curve.cpu().numpy()
            if isinstance(js_curve, torch.Tensor)
            else js_curve
        )

        refined = [0]
        T = len(js_curve)

        for b in boundaries[1:-1]:
            left = max(0, b - search_radius)
            right = min(T, b + search_radius + 1)

            local_peak = np.argmax(js_curve[left:right])
            refined.append(left + local_peak)

        refined.append(boundaries[-1])
        refined = sorted(set(refined))
        return refined

    @staticmethod
    def compute_js_curve(logits, top_k=100):
        """
        计算相邻帧JS散度变化曲线
        仅保留Top-K类别降低噪声
        """

        prob = F.softmax(logits, dim=1)

        # 只保留Top-K概率
        top_values, _ = torch.topk(prob, k=min(top_k, prob.shape[1]), dim=1)

        # 重新归一化
        top_values = top_values / (top_values.sum(dim=1, keepdim=True) + 1e-8)

        T = top_values.shape[0]
        if T <= 1:
            return torch.zeros(T, device=logits.device)

        js_scores = []
        eps = 1e-8

        for t in range(T - 1):
            p = top_values[t]
            q = top_values[t + 1]
            m = 0.5 * (p + q)

            kl_pm = torch.sum(p * torch.log((p + eps) / (m + eps)))
            kl_qm = torch.sum(q * torch.log((q + eps) / (m + eps)))
            js = 0.5 * (kl_pm + kl_qm)
            js_scores.append(js)

        js_scores = torch.stack(js_scores)

        js_scores = torch.cat([js_scores, js_scores[-1:].clone()], dim=0)

        return js_scores

    @staticmethod
    def smooth_curve(curve, sigma=1):
        """
        高斯平滑
        """

        if isinstance(curve, torch.Tensor):
            device = curve.device
            curve_np = curve.detach().cpu().numpy()
            smooth_np = gaussian_filter1d(curve_np, sigma=sigma)
            return torch.tensor(smooth_np, dtype=curve.dtype, device=device)

        return gaussian_filter1d(curve, sigma=sigma)

    @staticmethod
    def adaptive_clip_by_segments(scores, min_segment_len=2, js_top_k=20, js_search_radius=2):
        """
        基于Top1类别连续区域的自适应切分

        参数:
            scores: List[(T,V)]
            min_segment_len: 小于该长度的类别段视为抖动
            js_top_k: JS计算时保留Top-K类别
            js_search_radius: JS局部搜索窗口

        返回:
            clip_boundaries: List[List[int]]
        """
        clip_boundaries = []
        for b in range(len(scores)):
            s = scores[b]  # (T, V)
            print(f"[Auto Debug] 视频{b}: score_matrix shape={s.shape}")
            T = len(s)
            if T <= 1:
                clip_boundaries.append([0, T])
                continue

            # Step1 获取Top1预测
            pred = torch.argmax(s, dim=1)
            print(f"[Auto Debug] 视频{b}: "f"原始Top1序列={pred.tolist()}")

            # Step2 平滑短时抖动
            pred = ModelInferenceUtil.smooth_prediction_segments(pred, min_segment_len=min_segment_len)
            print(f"[Auto Debug] 视频{b}: "f"平滑后Top1序列={pred.tolist()}")

            # Step3 提取边界
            boundaries = ModelInferenceUtil.extract_change_boundaries(pred)

            print(f"[Auto Debug] 视频{b}: "f"类别变化边界={boundaries}")

            # Step4 JS散度校正
            js_curve = ModelInferenceUtil.compute_js_curve(s, top_k=js_top_k)
            js_curve = ModelInferenceUtil.smooth_curve(js_curve, sigma=1)
            refined_boundaries = (
                ModelInferenceUtil.refine_boundaries_with_js(
                    boundaries,
                    js_curve,
                    js_search_radius
                )
            )

            # Step5 去重排序
            refined_boundaries = sorted(set(refined_boundaries))

            # 防止出现极短片段
            final_boundaries = [refined_boundaries[0]]

            for idx in refined_boundaries[1:]:
                if idx - final_boundaries[-1] >= min_segment_len:
                    final_boundaries.append(idx)

            if final_boundaries[-1] != T:
                final_boundaries.append(T)

            print(f"[Auto Debug] 视频{b}: "f"最终边界={final_boundaries}")
            clip_boundaries.append(final_boundaries)

        return clip_boundaries

    @staticmethod
    def group_scores_by_boundaries(scores, clip_boundaries, top_k):
        """
        根据边界分组 scores，并返回每组的 Top-K 以及帧区间
        
        参数:
            scores: List[Tensor], 每个元素是 (T, V)
            clip_boundaries: List[List[int]], 每个视频的边界
            top_k: 返回每组前k个
        
        返回:
            clip_info: List[List[Dict[str, Any]]], 每组包含 keywords 与 start/end
        """
        clip_info = []
        for b in range(len(scores)):
            s = scores[b]
            boundaries = clip_boundaries[b]
            clip_results = []

            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                clip_scores = s[start:end]  # (clip_len, V)
                # 该clip所有帧求平均
                mean_scores = torch.mean(clip_scores, dim=0)  # (V,)
                # prob = F.softmax(clip_scores, dim=1)
                # mean_scores = prob.mean(dim=0)
                # 取Top-K
                topk_indices = torch.topk(mean_scores, k=top_k).indices
                clip_results.append({
                    "indices": topk_indices,
                    "start": start,
                    "end": end,
                })

            clip_info.append(clip_results)

        return clip_info

    @staticmethod
    @torch.no_grad()
    def kws_inference(video_lip, video_mask, device="cpu", strategy="Manu", clip=1, top_k=5):
        # model
        model = WPELipReader()
        if strategy == "Manu":
            model = ModelInferenceUtil.load_model_state(
                model,
                args.path_wpelip_hz,
                device,
            )
        elif strategy == "Auto":
            # Auto策略：使用相同的模型，通过自适应分组实现
            model = ModelInferenceUtil.load_model_state(
                model,
                args.path_wpelip_hz,
                device,
            )
        # cpu --> gpu
        model = model.to(device)
        video_lip = video_lip.to(device)
        video_mask = video_mask.to(device)
        # extract feature
        c3d_feature = ModelInferenceUtil.extract_feature(video_lip, device)
        # inference
        model.eval()
        scores = model.kws_inference(c3d_feature, video_mask)

        # data post process
        clip_info_batch = []
        if strategy == "Manu":
            # Manu策略：固定clip分组
            clip_group = (len(scores[0]) + clip - 1) // clip
            for b in range(len(scores)):
                clip_results = []
                for c in range(clip_group):
                    start_idx = c * clip
                    end_idx = min((c + 1) * clip, len(scores[0]))
                    data = scores[b][start_idx:end_idx]
                    data_mean = torch.mean(data, dim=0, keepdim=True).squeeze(0)
                    topk_indices = torch.topk(data_mean, k=top_k).indices
                    clip_results.append({
                        "indices": topk_indices,
                        "start": start_idx,
                        "end": end_idx,
                    })
                clip_info_batch.append(clip_results)
        else:
            # Auto策略：基于帧级类别连续区域的自适应分组
            clip_boundaries = ModelInferenceUtil.adaptive_clip_by_segments(scores)
            clip_info_batch = ModelInferenceUtil.group_scores_by_boundaries(scores, clip_boundaries, top_k)

        # decode text
        predict_keyword = ModelInferenceUtil.kws_idx_to_hz(clip_info_batch)
        return predict_keyword

    @staticmethod
    @torch.no_grad()
    def lipreading_inference(video_lip, video_mask, device="cpu", language="hz"):
        # model
        model = WPELipReader()
        if language == "hz":
            model = ModelInferenceUtil.load_model_state(
                model,
                args.path_wpelip_hz,
                device,
            )
        elif language == "py":
            model = ModelInferenceUtil.load_model_state(
                model,
                args.path_wpelip_py,
                device,
            )
        # cpu --> gpu
        model = model.to(device)
        video_lip = video_lip.to(device)
        video_mask = video_mask.to(device)
        # extract feature
        c3d_feature = ModelInferenceUtil.extract_feature(video_lip, device)
        # inference
        model.eval()
        ic(c3d_feature.device)
        ic(next(model.parameters()).device)
        scores, logits, predict = model(c3d_feature, video_mask)
        predict_text = []
        if language == "hz":
            predict_text = ModelInferenceUtil.idx_to_hz(predict)
        elif language == "py":
            predict_text = ModelInferenceUtil.idx_to_py(predict)
        # return
        return predict_text

    @staticmethod
    @torch.no_grad()
    def extract_feature(lips, device="cpu"):
        c3d_model = C3dModel(decode_mode="hz")
        model = ModelInferenceUtil.load_model_state(
            c3d_model,
            args.path_c3d,
            device,
        )
        model.eval()
        model = model.to(device)
        return model.cnn.forward(lips)

    @staticmethod
    def load_model_state(model, ckpt_file, device):
        """
        从 checkpoint 中加载 model
        """
        checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def prepare_data(lip):
        video_lip, video_mask = [], []
        for lip in lip:
            # norm
            lip = list(filter(lambda im: not im is None, lip))
            lip = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in lip]
            lip = np.stack(lip, axis=0).astype(np.float32)
            lip = ModelInferenceUtil.norm(lip, [0.4379, 0.4964, 0.6584], [0.1218, 0.1406, 0.1649])
            # mask
            lip_mask = torch.zeros(lip.shape[0], dtype=torch.int)
            # padding
            lip = ModelInferenceUtil.pad_sequence(lip, 230)
            lip_mask = ModelInferenceUtil.pad_sequence(lip_mask, 230, 1)
            # numpy --> tensor
            lip = torch.from_numpy(lip).float()
            lip = lip.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
            lip_mask = torch.from_numpy(lip_mask).bool()
            # to list
            video_lip.append(lip)
            video_mask.append(lip_mask)
        # to torch tensor
        video_lip = torch.stack(video_lip)
        video_mask = torch.stack(video_mask)
        return video_lip, video_mask

    @staticmethod
    def idx_to_hz(array):
        res = list()
        for b in range(array.shape[0]):
            element = []
            for n in array[b]:
                if n == ModelInferenceUtil.eos_token:
                    break
                if (
                    n != ModelInferenceUtil.pad_token
                    and n != ModelInferenceUtil.blank_token
                    and n != ModelInferenceUtil.sos_token
                ):
                    element.append(ModelInferenceUtil.hanzi_vocab[n.item()])
            res.append(" ".join(element).strip())
        return res

    @staticmethod
    def idx_to_py(array):
        res = list()
        for b in range(array.shape[0]):
            element = []
            for n in array[b]:
                if n == ModelInferenceUtil.eos_token:
                    break
                if n != ModelInferenceUtil.pad_token and n != ModelInferenceUtil.blank_token:
                    element.append(ModelInferenceUtil.pinyin_vocab[n.item()])
            res.append(" ".join(element).strip() + " <eos>")
        return res

    @staticmethod
    def kws_idx_to_hz(array):
        res_batch = list()
        for b in range(len(array)):
            res = list()
            for g in range(len(array[b])):
                info = array[b][g]
                element = []
                for n in info["indices"]:
                    element.append(ModelInferenceUtil.hanzi_vocab[n.item()])
                res.append({
                    "keywords": element,
                    "start": info["start"],
                    "end": info["end"],
                })
            res_batch.append(res)
        return res_batch

    @staticmethod
    def kws_idx_to_py(array):
        res_batch = list()
        for b in range(len(array)):
            res = list()
            for g in range(len(array[0])):
                element = []
                for n in array[b][g]:
                    element.append(ModelInferenceUtil.pinyin_vocab[n.item()])
                res.append(element)
            res_batch.append(res)
        return res_batch

    @staticmethod
    def norm(img, mean, std):
        img = img / 255.0
        img = (img - mean) / std
        return img

    @staticmethod
    def pad_sequence(data, max_length, pad_value=0):
        data = [data[i] for i in range(data.shape[0])]
        size = data[0].shape
        for i in range(max_length - len(data)):
            data.append(np.full(size, fill_value=pad_value))
        return np.stack(data)
