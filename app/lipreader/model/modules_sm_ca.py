import torch.nn as nn


class SimilarityMatchLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(SimilarityMatchLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_mask):
        attn_out, _ = self.self_attention.forward(q, kv, kv, key_padding_mask=kv_mask)
        q = self.layer_norm1(q + self.dropout(attn_out))

        ff_out = self.feed_forward(q)
        q = self.layer_norm2(q + self.dropout(ff_out))

        return q


class SimilarityMatch(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        num_layers,
    ):
        super(SimilarityMatch, self).__init__()

        self.layers = nn.ModuleList(
            [
                SimilarityMatchLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, q, kv, kv_mask=None):
        for layer in self.layers:
            q = layer(q, kv, kv_mask)
        return q


class CrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(CrossEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_mask):
        attn_out, _ = self.self_attention.forward(q, kv, kv, key_padding_mask=kv_mask)
        q = self.layer_norm1(q + self.dropout(attn_out))

        ff_out = self.feed_forward(q)
        q = self.layer_norm2(q + self.dropout(ff_out))

        return q


class CrossEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        num_layers,
    ):
        super(CrossEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [CrossEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

    def forward(self, q, kv, kv_mask=None):
        for layer in self.layers:
            q = layer(q, kv, kv_mask)
        return q
