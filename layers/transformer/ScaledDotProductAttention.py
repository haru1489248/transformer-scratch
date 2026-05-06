import numpy as np
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        '''
        d_k: 各単語のベクトル（分散表現）の次元数
        '''
        super().__init__()
        self.d_k = d_k

    def forward(
            self,
            q: torch.Tensor, # =Q
            k: torch.Tensor, # =X
            v: torch.Tensor, # =X
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        '''
        q, k shape = (head * batch_size, seq_len, d_k) v shape = (head * batch_size, seq_len, d_v)
        q: self-attentionの場合入力Xとなる。cross-attentionの場合Decoderの前の層の出力
        k: self-attentionの場合入力Xとなる。cross-attentionの場合Encoderの出力
        v: kと同じ
        mask: paddingなどでAttentionスコアを0にしたい位置をマスクするためのもの
        '''
        scalar = np.sqrt(self.d_k)
        # torch.transpose()は指定した位置の次元を転置する
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar # Q * X^T / (D^0.5) を計算

        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            # masked_fill_()は第一引数がTrueの位置に対して第二引数を代入する
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max # softmaxに通すと-infは実質0と同じとみなせる
            )

        attention_weight = nn.functional.softmax(attention_weight, dim=2) # Attention weightを計算
        return torch.matmul(attention_weight, v) # (Attention weight) * X により重みづけ
