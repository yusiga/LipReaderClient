import torch
import torch.nn as nn

from .modules import Seq2Seq
from .modules_sm_ca import SimilarityMatch, CrossEncoder


class WPELipReader(nn.Module):
    def __init__(
        self,
        decode_mode="hz",
        d_model=512,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.5,
        cross_num_layers=6,
    ):
        super().__init__()
        if decode_mode == "py":
            self.lexicon_size = 417
        elif decode_mode == "hz":
            self.lexicon_size = 3521
        self.top_k = 1

        # self.cnn = Cnn3d()

        # * Similarity Match
        self.sm = SimilarityMatch(d_model, nhead, dim_feedforward, dropout, cross_num_layers)

        # * 映射词库上
        self.fc_lexicon = nn.Linear(d_model, self.lexicon_size)

        # * Cross Attention
        self.ca_1 = CrossEncoder(d_model, nhead, dim_feedforward, dropout, cross_num_layers)
        self.ca_2 = CrossEncoder(d_model, nhead, dim_feedforward, dropout, cross_num_layers)

        # * Seq2Seq
        self.seq2seq = Seq2Seq(vocab_size=self.lexicon_size)

    def forward(self, video, video_mask, tgt=None):
        B = video.shape[0]
        length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")
        # video = self.cnn.forward(video)
        # * sm
        lexicon = [torch.arange(0, self.lexicon_size) for _ in range(B)]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.seq2seq.decoder.embedding(lexicon)  # shape: (B, V, d_model)

        scores = self.sm(video, lexicon)
        scores = self.fc_lexicon(scores)

        word = torch.topk(scores, k=self.top_k)[1].to(video.device)  # (B, T, top_k)
        word = self.seq2seq.decoder.embedding(word)  # (B, T, top_k, d_model)
        word = torch.sum(word, dim=2)  # (B, T, d_model)

        ca_1 = self.ca_1(video, word, video_mask)
        ca_2 = self.ca_2(word, video, video_mask)
        fusion = ca_1 + ca_2

        if tgt is not None:
            logits, pred = self.seq2seq.forward(fusion, length, tgt)
        else:
            logits, pred = self.seq2seq.greedy_search(fusion, length, 30)
        return scores, logits, pred

    def inference(self, video, video_mask, beam_size):
        B = video.shape[0]
        length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")
        # video = self.cnn.forward(video)
        # * sm
        lexicon = [torch.arange(0, self.lexicon_size) for _ in range(B)]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.seq2seq.decoder.embedding(lexicon)  # shape: (B, V, d_model)

        scores = self.sm(video, lexicon)
        scores = self.fc_lexicon(scores)

        word = torch.topk(scores, k=self.top_k)[1].to(video.device)  # (B, T, top_k)
        word = self.seq2seq.decoder.embedding(word)  # (B, T, top_k, d_model)
        word = torch.sum(word, dim=2)  # (B, T, d_model)

        ca_1 = self.ca_1(video, word, video_mask)
        ca_2 = self.ca_2(word, video, video_mask)
        fusion = ca_1 + ca_2

        # * 解码
        return self.seq2seq.beam_search(fusion, length, 30, beam_size)

    def kws_inference(self, video, video_mask):
        B = video.shape[0]
        length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")
        # video = self.cnn.forward(video)
        # * sm
        lexicon = [torch.arange(0, self.lexicon_size) for _ in range(B)]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.seq2seq.decoder.embedding(lexicon)  # shape: (B, V, d_model)

        scores = self.sm(video, lexicon)
        scores = self.fc_lexicon(scores)  # (B, T, V)
        scores = [score[:l] for score, l in zip(scores, length)]

        # word = torch.topk(scores, k=top_k)[1].to(video.device)  # (B, T, top_k)
        # word = self.seq2seq.decoder.embedding(word)  # (B, T, top_k, d_model)
        # word = torch.sum(word, dim=2)  # (B, T, d_model)
        #
        # ca_1 = self.ca_1(video, word, video_mask)
        # ca_2 = self.ca_2(word, video, video_mask)
        # fusion = ca_1 + ca_2

        # if tgt is not None:
        #     logits, pred = self.seq2seq.forward(fusion, length, tgt)
        # else:
        #     logits, pred = self.seq2seq.greedy_search(fusion, length, 30)
        # return scores, logits, pred
        return scores


class Cnn3d(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.bn3 = nn.BatchNorm3d(96)

        self.gru1 = nn.GRU(3072, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self._init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.permute(2, 0, 1, 3, 4).contiguous()  # (T, B, 96, 4, 8)
        x = x.view(x.size(0), x.size(1), -1)  # (T, B, 3072)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = x.permute(1, 0, 2).contiguous()
        return x

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
