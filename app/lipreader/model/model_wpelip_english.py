import torch
import torch.nn as nn

from .modules import Seq2Seq
from .modules_sm_ca import SimilarityMatch, CrossEncoder


class LipReader(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.5,
        cross_num_layers=6,
        word_lexicon_size=17851,
        phone_lexicon_size=74,
        top_k=1,
    ):
        super().__init__()
        self.word_lexicon_size = word_lexicon_size
        self.phone_lexicon_size = phone_lexicon_size
        self.top_k = top_k

        self.phone_embedding = nn.Embedding(self.phone_lexicon_size, d_model)

        # * Similarity Match
        self.sm = SimilarityMatch(d_model, nhead, dim_feedforward, dropout, cross_num_layers)

        # * 映射词库上
        self.fc_lexicon = nn.Linear(d_model, self.phone_lexicon_size)
        self.fc_word_lexicon = nn.Linear(d_model, self.word_lexicon_size)

        # * Cross Attention
        self.ca_1 = CrossEncoder(d_model, nhead, dim_feedforward, dropout, cross_num_layers)
        self.ca_2 = CrossEncoder(d_model, nhead, dim_feedforward, dropout, cross_num_layers)

        # * Seq2Seq
        self.seq2seq = Seq2Seq(vocab_size=self.word_lexicon_size)

    def forward(self, video, video_mask, tgt=None):
        B = video.shape[0]
        length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")
        # video = self.cnn.forward(video)
        # * sm
        lexicon = [torch.arange(0, self.phone_lexicon_size) for _ in range(B)]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.phone_embedding(lexicon)  # shape: (B, V, d_model)

        scores = self.sm(video, lexicon)
        scores_phone = self.fc_lexicon(scores)
        scores_word = self.fc_word_lexicon(scores)

        word = torch.topk(scores_phone, k=self.top_k)[1].to(video.device)  # (B, T, top_k)
        word = self.phone_embedding(word)  # (B, T, top_k, d_model)
        word = torch.sum(word, dim=2)  # (B, T, d_model)

        ca_1 = self.ca_1(video, word, video_mask)
        ca_2 = self.ca_2(word, video, video_mask)
        fusion = ca_1 + ca_2

        if tgt is not None:
            logits, pred = self.seq2seq.forward(fusion, length, tgt)
        else:
            logits, pred = self.seq2seq.greedy_search(fusion, length, 25)
        return scores_phone, scores_word, logits, pred

    def lipreading_inference(self, video, video_mask):
        B = video.shape[0]
        length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")
        # video = self.cnn.forward(video)
        # * sm
        lexicon = [torch.arange(0, self.phone_lexicon_size) for _ in range(B)]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.phone_embedding(lexicon)  # shape: (B, V, d_model)

        scores = self.sm(video, lexicon)
        scores_phone = self.fc_lexicon(scores)
        scores_word = self.fc_word_lexicon(scores)

        word = torch.topk(scores_phone, k=self.top_k)[1].to(video.device)  # (B, T, top_k)
        word = self.phone_embedding(word)  # (B, T, top_k, d_model)
        word = torch.sum(word, dim=2)  # (B, T, d_model)

        ca_1 = self.ca_1(video, word, video_mask)
        ca_2 = self.ca_2(word, video, video_mask)
        fusion = ca_1 + ca_2

        logits, pred = self.seq2seq.greedy_search(fusion, length, 25)
        return pred

    def kws_inference(self, video, video_mask):
        B = video.shape[0]
        length = torch.sum(1 - video_mask.long(), dim=-1).to("cpu")
        # * sm
        lexicon = [torch.arange(0, self.phone_lexicon_size) for _ in range(B)]
        lexicon = torch.stack(lexicon).to(video.device)
        lexicon = self.phone_embedding(lexicon)  # shape: (B, V, d_model)

        scores = self.sm(video, lexicon)
        scores_word = self.fc_word_lexicon(scores)  # (B, T, V)
        scores_result = [score[:l] for score, l in zip(scores_word, length)]

        # word = torch.topk(scores_phone, k=self.top_k)[1].to(video.device)  # (B, T, top_k)
        # word = self.phone_embedding(word)  # (B, T, top_k, d_model)
        # word = torch.sum(word, dim=2)  # (B, T, d_model)

        # ca_1 = self.ca_1(video, word, video_mask)
        # ca_2 = self.ca_2(word, video, video_mask)
        # fusion = ca_1 + ca_2

        # if tgt is not None:
        #     logits, pred = self.seq2seq.forward(fusion, length, tgt)
        # else:
        #     logits, pred = self.seq2seq.greedy_search(fusion, length, 25)
        return scores_result
