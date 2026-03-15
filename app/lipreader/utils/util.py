from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
import face_alignment
import os

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
    @torch.no_grad()
    def kws_inference(video_lip, video_mask, device="cpu", strategy="Manu", clip=1, top_k=5):
        # model
        model = WPELipReader()
        if strategy == "Manu":
            model = ModelInferenceUtil.load_model_state(
                model,
                # "/data/LP2/LipReaderClient/app/lipreader/checkpoint/checkpoint_wpelip_hz.pth",
                args.path_wpelip_hz,
                # "H:\\LipReaderClient\\app\\lipreader\\checkpoint\\checkpoint_wpelip_hz.pth",
                device,
            )
        elif strategy == "Auto":
            model = ModelInferenceUtil.load_model_state(
                # TODO
                model,
                # "/data/LP2/LipReaderClient/app/lipreader/checkpoint/checkpoint_wpelip_py.pth",
                args.path_wpelip_hz,
                # "H:\\LipReaderClient\\app\\lipreader\\checkpoint\\checkpoint_wpelip_hz.pth",
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
        clip_group = (len(scores[0]) + clip - 1) // clip
        new_scores = []
        for b in range(len(scores)):
            new_lst = []
            for c in range(clip_group):
                start_idx = c * clip
                end_idx = min((c + 1) * clip, len(scores[0]))
                data = scores[b][start_idx:end_idx]
                data_mean = torch.mean(data, dim=0, keepdim=True)
                new_lst.append(data_mean.squeeze(0))
                # new_scores[b][c] = data_mean.squeeze(0)
            new_scores.append(torch.stack(new_lst))
        # keywords = torch.topk(new_scores, k=top_k, dim=-1)[1]
        keywords = [torch.topk(score, k=top_k, dim=-1)[1] for score in new_scores]
        # decode text
        # ic(keywords)
        predict_keyword = []
        predict_keyword = ModelInferenceUtil.kws_idx_to_hz(keywords)
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
            for g in range(len(array[0])):
                element = []
                for n in array[b][g]:
                    element.append(ModelInferenceUtil.hanzi_vocab[n.item()])
                res.append(element)
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
