import torch
from torchtext.data.metrics import bleu_score
from torchtext.vocab import Vocab
from utils.text.vocab import BOS, EOS, PAD, UNK

class BleuScore:
    def __init__(
            self,
            tgt_vocab: Vocab,
            ignore_tokens: list[str] = [
                PAD,
                UNK,
                EOS,
                BOS,
            ],
    ) -> None:
        self.tgt_vocab = tgt_vocab
        # 無視するものを単語idで保存
        self.ignore_token_indices = self.tgt_vocab.lookup_indices(tokens=ignore_tokens)

    def __call__(self, output: torch.Tensor, tgt: torch.Tensor) -> float:
        output_list = output.tolist()
        tgt_list = tgt.tolist()

        candidates: list[list[str]] = []
        for item in output_list:
            candidates.append(
                [
                    self.tgt_vocab.lookup_token(i)
                    for i in item
                    if i not in self.ignore_token_indices
                ]
            )

        # 一つのソースに対してターゲットは複数ある（正解翻訳が複数ある）ことを想定して二重のlistにしている
        references: list[list[list[str]]] = []
        for item in tgt_list:
            references.append(
                [
                    [
                        self.tgt_vocab.lookup_token(i)
                        for i in item
                        if i not in self.ignore_token_indices
                    ]
                ]
            )
        return bleu_score(candidates, references)
