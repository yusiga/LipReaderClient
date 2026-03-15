import numpy as np
import torch
import torch.nn as nn
from icecream import ic


class AutoKws(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.decode_mode == "py":
            self.lexicon_size = args.py_lexicon_size
        elif args.decode_mode == "hz":
            self.lexicon_size = args.hz_lexicon_size

        self.embedding = nn.Embedding(self.lexicon_size, args.d_model)

        self.ca = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                batch_first=True,
                norm_first=True,
            ),
            args.num_layers,
        )

        self.frame_classifier = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 2),
            nn.ReLU(),
            nn.Linear(args.d_model * 2, self.lexicon_size),
        )
        self.boundary_detector = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Linear(args.d_model // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.text_classifier = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 2),
            nn.ReLU(),
            nn.Linear(args.d_model * 2, self.lexicon_size),
        )

    def forward(self, video, video_mask):
        # length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")

        # lexicon
        lexicon = [torch.arange(0, self.lexicon_size) for _ in range(video.size(0))]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.embedding(lexicon)  # shape: (B, V, d_model)

        video = self.ca.forward(tgt=video, memory=lexicon, tgt_key_padding_mask=video_mask)

        frame_logits = self.frame_classifier(video)
        boundary_logits = self.boundary_detector(video)
        text_feature = self.aggregate_clips(boundary_logits, video)
        text_logits = self.text_classifier(text_feature)

        return text_logits, frame_logits, boundary_logits.squeeze(-1)

    def aggregate_clips(self, boundary_prediction, video_feature):
        """
        根据tensor1中概率高于0.5的位置确定clip边界，对tensor2相应clip频段中的特征进行聚合。
        参数:
        tensor1: 形状为 (B, T, 1) 的Tensor，表示每一帧的概率，值在0~1之间
        tensor2: 形状为 (B, T, D) 的Tensor，表示每一帧的特征向量
        返回:
        aggregated_features: 形状为 (B, ?, D) 的Tensor，? 小于 T，聚合后的特征
        """
        boundary_prediction = self.sigmoid(boundary_prediction)
        self.device = boundary_prediction.device
        B, T, _ = boundary_prediction.size()
        clip_boundaries = []
        for b in range(B):
            start = None
            clip_list = []
            for t in range(T):
                if boundary_prediction[b, t, 0] >= torch.max(boundary_prediction[b]) * 0.3:  #
                    clip_list.append(t)
            if clip_list[0] != 0:
                clip_list.insert(0, 0)
            if len(clip_list) == 1:
                clip_list.append(len(boundary_prediction[b]))
            clip_boundaries.append(clip_list)

        aggregated_features = []
        for b in range(B):
            feature_list = []
            for i in range(len(clip_boundaries[b]) - 1):
                start, end = clip_boundaries[b][i], clip_boundaries[b][i + 1]
                clip_features = video_feature[b, start:end]  # 获取clip对应的特征片段 (clip, D)
                aggregated_clip_feature = torch.mean(clip_features, dim=0)  # (D)
                feature_list.append(aggregated_clip_feature)  # (1, ?, D)
            if len(feature_list) == 0:
                ic(clip_boundaries[b])
                ic(boundary_prediction[b])
            aggregated_features.append(torch.stack(feature_list))  # (B, ?, D)
        feature_len = [len(f) for f in aggregated_features]
        max_len = max(feature_len)

        new_features = [self._pad_sequence(feature, max_len, 0) for feature in aggregated_features]
        return torch.stack(new_features).to(self.device)

    def _pad_sequence(self, data, max_length, pad_value=0):
        data = [data[i] for i in range(data.shape[0])]
        size = data[0].shape
        for i in range(max_length - len(data)):
            data.append(torch.full(size, fill_value=pad_value, device=self.device))
        return torch.stack(data)
