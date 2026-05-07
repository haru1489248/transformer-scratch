import numpy as np
import torch
from torch import nn

class AddPositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            max_len: int,
            device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        # あらかじめ用意された位置情報ベクトルからseq_len分抜き出す
        # batch_sizeが0次元目に入っていることを考慮して、positional_encoding_weightも0次元目にバッチサイズの次元を作る
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        # 2 * (i // 2) によって、偶数・奇数のペアで同じ周波数を作る
        # d_model で割ることで、次元に応じた周波数のスケールを調整する
        # 一意の位置情報を単語の分散表現の中に埋め込むことができる
        w = pos / (10000 ** ((2 * (i // 2)) / self.d_model))

        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
        [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
        for pos in range(1, self.max_len + 1)
        ]
