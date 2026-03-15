import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random


class Seq2Seq(nn.Module):
    def __init__(
            self,
            vocab_size,
            enc_input_size=512,
            enc_hidden_size=256,
            dec_input_size=512,
            dec_hidden_size=512,
            n_layers=2,
            dropout=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.enc_hidden_size = enc_hidden_size
        self.n_layers = n_layers

        # self.cnn = Cnn3d()
        self.encoder = PackedEncoder(enc_input_size, enc_hidden_size, n_layers, dropout)
        self.decoder = Decoder(vocab_size, dec_input_size, dec_hidden_size, n_layers, dropout)

    def forward(self, video, length, target):
        """
        @param video: (B, 3, T, H, W)
        @param length: T
        @param target: (L, B)
        """
        video = video.permute(1, 0, 2).contiguous()
        target = target.permute(1, 0).contiguous()
        L, B = target.shape
        # features = self.cnn.forward(video)  # (T, B, 512)

        # (T, B, 512), (n_layers * 2, B, 256)
        enc_output, enc_hidden = self.encoder.forward(video, length)

        dec_hidden = enc_hidden.permute(1, 0, 2).contiguous()  # (B, n_layers * 2, 256)
        dec_hidden = dec_hidden.view(
            -1, self.n_layers, self.enc_hidden_size * 2
        )  # (B, n_layers, 512)
        dec_hidden = dec_hidden.permute(1, 0, 2).contiguous()  # (n_layers, B, 512)

        dec_input = torch.ones(1, B, dtype=torch.long).to(video.device)
        output = torch.empty(L, B, self.vocab_size).to(video.device)
        predict = torch.empty(L, B).to(video.device)

        for i in range(L):
            dec_output, dec_hidden, alpha = self.decoder.forward(dec_input, dec_hidden, enc_output)
            output[i] = dec_output  # (1, B, vocab_size)
            top1 = torch.argmax(dec_output, dim=-1)
            predict[i] = top1  # (1, B)
            """scheduled sampling: https://arxiv.org/pdf/1506.03099.pdf"""
            teacher_force = random.random() < 0.5
            dec_input = target[i].view(1, -1) if teacher_force else top1
        output = output.permute(1, 0, 2).contiguous()
        predict = predict.permute(1, 0).contiguous()
        predict = predict.long()
        return output, predict

    def greedy_search(self, video, length, max_len):
        B = video.size(0)
        L = max_len
        # features = self.cnn.forward(video)  # (T, B, 512)
        video = video.permute(1, 0, 2).contiguous()

        # (T, B, 512), (n_layers * 2, B, 256)
        enc_output, enc_hidden = self.encoder.forward(video, length)

        dec_hidden = enc_hidden.permute(1, 0, 2).contiguous()  # (B, n_layers * 2, 256)
        dec_hidden = dec_hidden.view(
            -1, self.n_layers, self.enc_hidden_size * 2
        )  # (B, n_layers, 512)
        dec_hidden = dec_hidden.permute(1, 0, 2).contiguous()  # (n_layers, B, 512)

        dec_input = torch.ones(1, B, dtype=torch.long).to(video.device)
        output = torch.empty(L, B, self.vocab_size).to(video.device)
        predict = torch.empty(L, B).to(video.device)

        for i in range(L):
            dec_output, dec_hidden, alpha = self.decoder.forward(dec_input, dec_hidden, enc_output)
            output[i] = dec_output  # (1, B, vocab_size)
            top1 = torch.argmax(dec_output, dim=-1)
            predict[i] = top1  # (1, B)
            dec_input = top1
        output = output.permute(1, 0, 2).contiguous()
        predict = predict.permute(1, 0).contiguous()
        predict = predict.long()
        return output, predict

    def beam_search(self, video, length, max_len, beam_size=2):
        B = video.size(0)
        L = max_len
        # features = self.cnn.forward(video)    # (T, B, 512)
        video = video.permute(1, 0, 2).contiguous()

        # (T, B, 512), (n_layers * 2, B, 256)
        enc_output, enc_hidden = self.encoder.forward(video, length)

        dec_hidden = enc_hidden.permute(1, 0, 2).contiguous()  # (B, n_layers * 2, 256)
        dec_hidden = dec_hidden.view(
            -1, self.n_layers, self.enc_hidden_size * 2
        )  # (B, n_layers, 512)
        dec_hidden = dec_hidden.permute(1, 0, 2).contiguous()  # (n_layers, B, 512)

        dec_hidden_list = []
        dec_input_list = []
        predict_list = []
        sos = torch.ones(1, B, dtype=torch.long).to(video.device)

        for i in range(L):
            if i == 0:
                # (1, B, vocab_size)
                dec_output, dec_hidden, alpha = self.decoder.forward(sos, dec_hidden, enc_output)
                _, topk = torch.topk(dec_output.squeeze(), k=beam_size)

                for j in range(beam_size):
                    dec_hidden_list.append(dec_hidden)
                    next = torch.ones(1, B, dtype=torch.long).fill_(topk[j]).to(video.device)
                    dec_input_list.append(next)
                    predict_list.append(next)
            else:
                topk = []
                for j in range(beam_size):
                    dec_input = dec_input_list[j]
                    dec_hidden = dec_hidden_list[j]
                    dec_output, dec_hidden, alpha = self.decoder.forward(
                        dec_input, dec_hidden, enc_output
                    )
                    dec_hidden_list[j] = dec_hidden
                    _, next = torch.max(dec_output, dim=-1)

                    topk.append(next.item())
                # prob = torch.cat(prob, dim=-1).squeeze()

                for j in range(beam_size):
                    next = torch.ones(1, B, dtype=torch.long).fill_(topk[j]).to(video.device)
                    # print(next)
                    dec_input_list[j] = next
                    predict_list[j] = torch.cat([predict_list[j], next], dim=0)

        return predict_list


class PackedEncoder(nn.Module):
    def __init__(self, enc_input_size=512, enc_hidden_size=256, n_layers=2, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(enc_input_size, enc_hidden_size, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self._init()

    def forward(self, input, length):
        """
        @param input: (T, B, 512)

        @return output_unpacked: (T, B, 2 * 256)
        @return hidden: (n_layer * 2, B, 256)
        """
        total_length = input.size(0)
        input_packed = pack_padded_sequence(input, length, enforce_sorted=False)

        self.gru.flatten_parameters()
        output_unpacked, hidden = self.gru(input_packed)
        output_unpacked, lens_unpacked = pad_packed_sequence(
            output_unpacked, total_length=total_length
        )

        return output_unpacked, hidden

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)


class Encoder(nn.Module):
    def __init__(self, enc_input_size=512, enc_hidden_size=256, n_layers=2, dropout=0.5):
        super().__init__()

        self.gru = nn.GRU(enc_input_size, enc_hidden_size, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self._init()

    def forward(self, input):
        """
        @param input: (T, B, 512)

        @return output: (T, B, 2 * 256)
        @return hidden: (n_layer * 2, B, 256)
        """
        self.gru.flatten_parameters()
        output, hidden = self.gru(input)

        return output, hidden

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)


class Decoder(nn.Module):
    def __init__(
            self, vocab_size, dec_input_size=512, dec_hidden_size=512, n_layers=2, dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dec_input_size)
        self.gru = nn.GRU(dec_input_size, dec_hidden_size, n_layers, bidirectional=False)

        self.attention = Attention()
        self.attention_fc = nn.Linear(dec_hidden_size * 2, dec_hidden_size)
        self.fc = nn.Linear(dec_hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self._init()

    def forward(self, input, hidden, enc_output):
        """
        @param input:      (1, B)
        @param hidden:     (2, B, 512)
        @param enc_output: (T, B, 512)

        @return output:    (1, B, vocab_size)
        @return hidden:    (2, B, 512)
        @return alpha:     (B, 1, T)
        """
        embedded = self.embedding(input)  # (1, B, 512)
        embedded = self.dropout(embedded)

        self.gru.flatten_parameters()
        # (1, B, 512), (2, B, 512)
        # print(.shape)
        # print(hidden.shape)
        output, hidden = self.gru(embedded, hidden)

        # (1, B, 512), (B, 1, T)
        context, alpha = self.attention.forward(hidden, enc_output)

        output = self.attention_fc(torch.cat((output, context), dim=2))
        output = self.relu(output)
        output = self.fc(output)  # (1, B, vocab_size)

        return output, hidden, alpha

    def _init(self):
        nn.init.kaiming_normal_(self.attention_fc.weight, nonlinearity="relu")
        nn.init.constant_(self.attention_fc.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)
        for name, param in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)


class Attention(nn.Module):
    """
    Reference: https://github.com/bentrevett/pytorch-seq2seq
    """

    def __init__(self, enc_hidden_size=256, dec_hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(enc_hidden_size * 4 + dec_hidden_size, dec_hidden_size)
        self.fc2 = nn.Linear(dec_hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self._init()

    def forward(self, hidden, enc_output):
        """
        @param hidden:      (2, B, 512)
        @param enc_output:  (T, B, 512)

        @return context:    (1, B, 512)
        @return alpha:      (B, 1, T)
        """
        T = enc_output.size(0)
        B = enc_output.size(1)

        hidden = hidden.permute(1, 0, 2).contiguous()  # (B, 2, 512)
        hidden = hidden.view(B, -1)  # (B, 1024)
        hidden = hidden.repeat(T, 1, 1)  # (T, B, 1024)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (B, T, 1024)
        enc_output = enc_output.permute(1, 0, 2).contiguous()  # (B, T, 512)
        concat = torch.cat((hidden, enc_output), dim=2)  # (B, T, 1536)

        alpha = self.tanh(self.fc1(concat))  # (B, T, 512)
        alpha = self.fc2(alpha)  # (B, T, 1)
        alpha = alpha.permute(0, 2, 1).contiguous()  # (B, 1, T)
        alpha = self.softmax(alpha)

        context = torch.bmm(alpha, enc_output)  # (B, 1, 512)
        context = context.permute(1, 0, 2).contiguous()  # (1, B, 512)

        return context, alpha

    def _init(self):
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.constant_(self.fc2.bias, 0)
