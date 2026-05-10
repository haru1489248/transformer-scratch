from typing import Generator, Optional
from torchtext.vocab import Vocab, build_vocab_from_iterator

PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
BOS = "<bos>"

def get_vocab(
        path_to_corpus: str,
        specials: list[str] = [
            PAD,
            UNK,
            EOS,
            BOS,
        ],
        vocab_size: Optional[int] = None,
) -> Vocab:
    # specials は先頭のインデックスに割り当てられるらしい
    # <pad>: 0, <unk>: 1, <eos>: 2, <bos>: 3
    return build_vocab_from_iterator(
        _yield_token(path_to_corpus), specials=specials, max_tokens=vocab_size
    )

def _yield_token(path_to_corpus: str) -> Generator[list[str], None, None]:
    with open(path_to_corpus, "r", encoding="utf-8") as f:
        for line in f:
            yield tokenize_sentence(line)

def tokenize_sentence(sentence: str) -> list[str]:
    """トークンごとに空白で区切られた文章をトークンの配列に変換する。"""
    return sentence.strip().split()
