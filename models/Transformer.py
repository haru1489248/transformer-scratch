import torch
from layers.transformer.TransformerDecoder import TransformerDecoder
from layers.transformer.TransformerEncoder import TransformerEncoder
from torch import nn

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            max_len: int,
            d_model: int = 512,
            heads_num: int = 8,
            d_ff: int = 2048,
            N: int = 6,
            dropout_rate: float = 0.1,
            layer_norm_eps: float = 1e-5,
            pad_idx: int = 0,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.N = N
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.pad_idx = pad_idx
        self.device = device

        self.encoder = TransformerEncoder(
            src_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        src : torch.Tensor
            単語のid列. [batch_size, max_len]
        tgt : torch.Tensor
            単語のid列. [batch_size, max_len]
        """

        # mask
        pad_mask_src = self._pad_mask(src)

        # src shape = (batch_size, seq_len, d_model)
        src = self.encoder(src, pad_mask_src)

        # target系列の"0(BOS)~max_len-1"(max_len-1系列)までを入力し、"1~max_len"(max_len-1系列)を予測する
        mask_self_attn = torch.logical_or(
            self._subsequent_mask(tgt), self._pad_mask(tgt)
        )
        dec_output = self.decoder(tgt, src, pad_mask_src, mask_self_attn)

        return self.linear(dec_output)

    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int
    ) -> torch.Tensor:
        """
        推論用のGreedy Decoding実装。
        1単語ずつ予測し、次のステップの入力に繋げる。

        Parameters:
        ----------
        src : torch.Tensor
            入力単語のID列 [batch_size, src_len]
        max_len : int
            生成する最大単語数
        bos_idx : int
            <bos> のID
        eos_idx : int
            <eos> のID

        Returns:
        -------
        torch.Tensor
            生成された単語のID列 [batch_size, generated_len]
        """
        batch_size = src.size(0)

        # 1. エンコード (これは1回だけで良い)
        pad_mask_src = self._pad_mask(src)
        memory = self.encoder(src, pad_mask_src)

        # 2. デコーダーの入力を <bos> で初期化 [batch_size, 1]
        # torch.full()は指定した形の行列を作成し、指定した値で全て埋める
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=self.device)

        for _ in range(max_len):
            # 現在の tgt に対してマスクを作成
            # torch.logical_or()は2つのテンソルの同じ位置にある要素同士を比較し、
            # どちらか一方がTrueならTrueという結果を返す関数
            mask_self_attn = torch.logical_or(
                self._subsequent_mask(tgt), self._pad_mask(tgt)
            )

            # 3. デコーダーで次の単語を予測
            dec_output = self.decoder(tgt, memory, pad_mask_src, mask_self_attn)
            output = self.linear(dec_output) # [batch_size, current_seq_len, tgt_vocab_size]

            # 最後のタイムステップの出力から、最も確率が高い単語を選択
            next_word = output[:, -1, :].argmax(dim=-1, keepdim=True) # [batch_size, 1]

            # 予測した単語を現在の tgt に結合して次の入力にする
            tgt = torch.cat([tgt, next_word], dim=1)

            # 全てのバッチが <eos> を出力したら終了
            if (next_word == eos_idx).all():
                break

        return tgt

    def beam_search(
            self,
            src: torch.Tensor,
            max_len: int,
            bos_idx: int,
            eos_idx: int,
            beam_size: int = 3,
        ) -> torch.Tensor:
            """
            Beam Searchによる推論。
            バッチサイズは1を想定している。
            """
            pad_mask_src = self._pad_mask(src)
            memory = self.encoder(src, pad_mask_src)

            # 各ビームの現在の系列と、その累積対数確率を保持
            # sequences: [beam_size, current_seq_len]
            sequences = torch.full((beam_size, 1), bos_idx, dtype=torch.long, device=self.device)
            # scores: 各ビームのスコア（対数確率の和）
            scores = torch.zeros(beam_size, device=self.device)

            # エンコード結果(memory)をビーム数分コピーしておく
            # memory: [beam_size, src_len, d_model]
            memory = memory.repeat(beam_size, 1, 1)
            pad_mask_src = pad_mask_src.repeat(beam_size, 1, 1)

            for _ in range(max_len):
                # 1. デコーダーで次の単語の確率分布を計算
                mask_self_attn = torch.logical_or(
                    self._subsequent_mask(sequences), self._pad_mask(sequences)
                )
                dec_output = self.decoder(sequences, memory, pad_mask_src, mask_self_attn)
                logits = self.linear(dec_output[:, -1, :]) # 最後の単語の予測のみ抽出

                # 2. 対数確率(log_softmax)に変換
                log_probs = torch.log_softmax(logits, dim=-1) # [beam_size, vocab_size]

                # 3. 累積スコアを計算 (現在のスコア + 新しい単語の対数確率)
                # 全ビーム×全単語の組み合わせ [beam_size, vocab_size]
                all_scores = scores.unsqueeze(1) + log_probs

                # 4. 上位 beam_size 個の候補を選択
                # flattenして全体からトップKを取得
                # 第二引数は取り出す個数
                top_k_scores, top_k_indices = torch.topk(all_scores.view(-1), beam_size)

                # 5. インデックスを「どのビームか」と「どの単語か」に分解
                beam_indices = top_k_indices // self.tgt_vocab_size
                token_indices = top_k_indices % self.tgt_vocab_size

                # 6. 次のステップのための sequences と scores を更新
                next_sequences = []
                next_scores = []

                for beam_idx, token_idx, score in zip(beam_indices, token_indices, top_k_scores):
                    # 1. どのビームから新しい単語が生成されたかを確認
                    # sequences[beam_idx]は、そのビームがこれまで作ってきた単語列を取り出す
                    prev_seq = sequences[beam_idx]

                    # 2. ビームの系列の末尾に、新しく選ばれた単語（token_idx）を結合する
                    next_sequences.append(torch.cat([prev_seq, token_idx.unsqueeze(0)]))
                    next_scores.append(score)

                # 次のdecoderに入力するsequences行列（batch_size, generated_seq_len)を作成する
                sequences = torch.stack(next_sequences)
                scores = torch.stack(next_scores)

                # 全てのビームが <eos> に到達したら終了（簡易版では省略可）
                if (sequences[:, -1] == eos_idx).all():
                    break

            # 最もスコアが高いビームを返す [1, seq_len]
            best_beam_idx = torch.argmax(scores)
            return sequences[best_beam_idx].unsqueeze(0)

    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """単語のid列(ex:[[4,1,9,11,0,0,0...],[4,1,9,11,0,0,0...],[4,1,9,11,0,0,0...]...])からmaskを作成する.
        Parameters:
        ----------
        x : torch.Tensor
            単語のid列. [batch_size, max_len]
        """
        seq_len = x.size(1)
        # eq()等しいかどうかを比較しbooleanで埋める
        mask = x.eq(self.pad_idx)  # 0 is <pad> in vocab
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)  # (batch_size, max_len, max_len)
        return mask.to(self.device)

    def _subsequent_mask(self, x: torch.Tensor) -> torch.Tensor:
        """DecoderのMasked-Attentionに使用するmaskを作成する.Causalと呼ばれる
        Parameters:
        ----------
        x : torch.Tensor
            単語のトークン列. [batch_size, max_len, d_model]
        """
        batch_size = x.size(0)
        max_len = x.size(1)
        # tril()は下三角行列を作成する。今回は下三角に1で埋めて他は0にし、最後に0のところをtrueにすることでmask行列を作成
        return (
            torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0).to(self.device)
        )
