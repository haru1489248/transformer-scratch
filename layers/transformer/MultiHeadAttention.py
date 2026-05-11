import torch
from layers.transformer.ScaledDotProductAttention import ScaledDotProductAttention
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        # ScaledDotProductで内積を求めるのでW_kとW_qの出力次元は同じにしないといけない
        self.W_k = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k) # ヘッド数, 入力次元, 出力次元（=入力次元/ヘッド数）
        )

        self.W_q = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k) # ヘッド数, 入力次元, 出力次元（=入力次元/ヘッド数）
        )
        self.W_v = nn.Parameter(
            torch.Tensor(h, d_model, self.d_v) # ヘッド数, 入力次元, 出力次元（=入力次元/ヘッド数）
        )

        # パラメータの初期値を設定する
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_v)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)
        # W shape = (d_model, h * d_v) b shape = (d_model)
        self.linear = nn.Linear(h * self.d_v, d_model)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask_3d: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        [Encoder]
        q, k, v = x (batch_size, src_seq_len, d_model)
        mask_3d = src_pad_mask (batch_size, seq_len, seq_len)
        [Decoder]
        q, k, v = x (batch_size, tgt_seq_len, d_model)
        mask_3d = tgt_pad_mask + tgt_causal_mask
        [Encoder-Decoder]
        k, v = encoder_output (batch_size, src_seq_len, d_model)
        q = decoder_input (batch_size, tgt_seq_len, d_model)
        mask_3d = src_pad_mask
        """
        batch_size = q.size(0)
        q_seq_len = q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        """repeat Query,Key,Value by num of heads"""
        # repeat()の1は元のサイズのまま維持する意味
        q = q.repeat(self.h, 1, 1, 1) # head, batch_size, q_seq_len, d_model
        k = k.repeat(self.h, 1, 1, 1) # head, batch_size, k_seq_len, d_model
        v = v.repeat(self.h, 1, 1, 1) # head, batch_size, v_seq_len, d_model

        """Linear before scaled dot product attention"""
        q = torch.einsum(
            "hijk,hkl->hijl", (q, self.W_q)
        ) # head, batch_size, q_seq_len, d_k
        k = torch.einsum(
            "hijk,hkl->hijl", (k, self.W_k)
        ) # head, batch_size, k_seq_len, d_k
        v = torch.einsum(
            "hijk,hkl->hijl", (v, self.W_v)
        ) # head, batch_size, v_seq_len, d_v

        """Split heads"""
        q = q.view(self.h * batch_size, q_seq_len, self.d_k)
        k = k.view(self.h * batch_size, k_seq_len, self.d_k)
        v = v.view(self.h * batch_size, v_seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.h, 1, 1)

        """Scaled dot product attention"""
        attention_output = self.scaled_dot_product_attention(
            q, k, v, mask_3d
        ) # (head*batch_size, q_seq_len, d_v)

        # torch.chunk()は指定したdimに沿って、第一引数を第二引数分の数に行列を分割する（返り値はTensorがh個入ったPythonのList）
        # attention_output = (batch_size, q_seq_len, d_v) * h個のリスト
        attention_output = torch.chunk(attention_output, self.h, dim=0)
        # torch.cat()は指定したdimに沿って、第一引数を連結する
        # attention_output shape = (batch_size, q_seq_len, d_v*h) = (batch_size, q_seq_len, d_model)
        attention_output = torch.cat(attention_output, dim=2)

        """Linear after scaled dot product attention"""
        # 1. 結合された情報の統合 (Information Mixing):
        #    torch.cat 直後の attention_output は、各ヘッドが独自の視点で計算した
        #    d_v 次元の「個別の答え」が物理的に横に並んでいるだけの状態。
        # 2. アフィン変換による「編集」プロセス:
        #    self.Linear (nn.Linear) を通すことで、各ヘッド間の情報を混ぜ合わせ、
        #    学習可能な重みを用いて、どのヘッドの情報を重視すべきかを調整する。
        # 3. 最終的な表現への昇華:
        #    ただの羅列を、次段の層が受け取れる「洗練された一つの文脈ベクトル」
        #    (d_model次元) へと統合・再構成して出力する
        output = self.linear(attention_output)
        return output
