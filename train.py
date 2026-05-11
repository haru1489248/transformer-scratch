from os.path import join

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from const.path import (
    FIGURE_PATH,
    KFTT_TOK_CORPUS_PATH,
    NN_MODEL_PICKLES_PATH,
    TANAKA_CORPUS_PATH,
)
from models import Transformer
from utils.dataset.Dataset import KfttDataset
from utils.evaluation.blue import BleuScore
from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import get_vocab

class Trainer:
    def __init__(
            self,
            net: nn.Module,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            bleu_score: BleuScore,
            device: torch.device,
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.bleu_score = bleu_score
        self.net = self.net.to(self.device)

    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds, labels)

    def train_step(
            self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.net.train()
        output = self.net(src, tgt)

        tgt = tgt[:, 1:] # decoderからの出力は1 ~ max_lenまでなので0以降のデータで誤差関数を計算する
        output = output[:, :-1, :] # 最後の文字の次の文字はないので捨てる

        # calculate loss
        # 直前でslice操作を行なっているのでcontiguous()を実行してメモリ上で連続になるようにしている
        # output shape = (batch_size * seq_len, vocab_size)
        # tgt shape = (batch_size * seq_len)
        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        # calculate bleu score
        # argmax()だとindexを返すがmax()だと要素そのものも返す
        _, output_ids = torch.max(output, dim=-1)
        bleu_score = self.bleu_score(output_ids, tgt)

        self.optimizer.zero_grad() # 前の勾配をリセットする
        loss.backward()
        self.optimizer.step()

        return loss, output, bleu_score

    def val_step(
            self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.net.eval()
        with torch.no_grad():
            output = self.net(src, tgt)

            tgt = tgt[:, 1:]
            output = output[:, :-1, :]

            loss = self.loss_fn(
                output.contiguous().view(
                    -1,
                    output.size(-1),
                ),
                tgt.contiguous().view(-1),
            )
            _, output_ids = torch.max(output, dim=-1)
            bleu_score = self.bleu_score(output_ids, tgt)

        return loss, output, bleu_score

    def fit(
            self, train_loader: DataLoader, val_loader: DataLoader, print_log: bool = True, epoch: int = 0, max_epoch: int = 0
    ) -> tuple[float, float, float, float]:
        # train
        train_losses: list[float] = []
        train_bleu_scores: list[float] = []
        if print_log:
            print(f"{'-'*20 + 'Train' + '-'*20} \n")
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _, bleu_score = self.train_step(src, tgt)

            # GPUのメモリを解放する
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(
                    f"train loss: {loss.item():.4f}, bleu score: {bleu_score:.4f}, " + f"iter: {i+1}/{len(train_loader)} epoch: {epoch}/{max_epoch} \n"
                )

            train_losses.append(loss.item())
            train_bleu_scores.append(bleu_score)

        # validation
        val_losses: list[float] = []
        val_bleu_scores: list[float] = []
        if print_log:
            print(f"{'-'*20 + 'Validation' + '-'*20} \n")
        for i, (src, tgt) in enumerate(val_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _, bleu_score = self.val_step(src, tgt)

            # 最初のバッチの時に翻訳サンプルを表示
            if i == 0:
                print(f"\n{'-'*10} Translation Samples {'-'*10}")
                print("[Greedy]")
                self.greedy_predict(src, tgt, num_samples=2)

                print("[Beam Search]")
                self.beam_search_predict(src, tgt, num_samples=2, beam_size=3)
                print(f"{'-'*40}\n")

            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(
                    f"val loss: {loss.item():.4f}, val bleu score: {bleu_score:.4f}, " + f"iter: {i+1}/{len(val_loader)} epoch: {epoch}/{max_epoch} \n"
                )

            val_losses.append(loss.item())
            val_bleu_scores.append(bleu_score)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_bleu = sum(train_bleu_scores) / len(train_bleu_scores)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_bleu = sum(val_bleu_scores) / len(val_bleu_scores)

        return avg_train_loss, avg_train_bleu, avg_val_loss, avg_val_bleu

    def greedy_predict(self, src: torch.Tensor, tgt: torch.Tensor, num_samples: int = 2):
        """
        Greedy Decodeを使用して翻訳結果を表示する
        """
        self.net.eval()
        with torch.no_grad():
            # bos_idx=3, eos_idx=2 (vocab.py の specials 定義順)
            output_ids = self.net.greedy_decode(
                src[:num_samples],
                max_len=24,
                bos_idx=3,
                eos_idx=2
            )

            for i in range(num_samples):
                # tensor_to_text はグローバルな src_vocab, tgt_vocab を参照
                src_text = src_tensor_to_text(src[i].cpu())
                tgt_text = tgt_tensor_to_text(tgt[i].cpu())
                pred_text = tgt_tensor_to_text(output_ids[i].cpu())

                print(f"Source:  {src_text}")
                print(f"Target:  {tgt_text}")
                print(f"Predict: {pred_text}")
                print("-" * 20)

    def beam_search_predict(self, src: torch.Tensor, tgt: torch.Tensor, num_samples: int = 1, beam_size: int = 3):
        """
        Beam Searchを使用して翻訳結果を表示する
        """
        self.net.eval()
        with torch.no_grad():
            # 1件ずつ処理（実装がバッチサイズ1想定のため）
            for i in range(num_samples):
                output_ids = self.net.beam_search(
                    src[i:i+1],
                    max_len=24,
                    bos_idx=3,
                    eos_idx=2,
                    beam_size=beam_size
                )

                src_text = src_tensor_to_text(src[i].cpu())
                tgt_text = tgt_tensor_to_text(tgt[i].cpu())
                pred_text = tgt_tensor_to_text(output_ids[0].cpu())

                print(f"Source: {src_text}")
                print(f"Target: {tgt_text}")
                print(f"Beam Search: {pred_text} (size: {beam_size})")
                print("-"*20)

    def test(self, test_data_loader: DataLoader) -> tuple[list[float], list[float]]:
        test_losses: list[float] = []
        test_bleu_scores: list[float] = []
        for i, (src, tgt) in enumerate(test_data_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _, bleu_score = self.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            test_losses.append(loss.item())
            test_bleu_scores.append(bleu_score)

        return test_losses, test_bleu_scores

if __name__ == "__main__":
    """
    1. define path and create vocab
    """
    TRAIN_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-train.en")
    TRAIN_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-train.ja")

    VAL_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-dev.en")
    VAL_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-dev.ja")

    TEST_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-test.en")
    TEST_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-test.ja")

    src_vocab = get_vocab(TRAIN_SRC_CORPUS_PATH, vocab_size=20000)
    tgt_vocab = get_vocab(TRAIN_TGT_CORPUS_PATH, vocab_size=20000)

    """
    2. define parameters
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    max_len = 24
    d_model = 128
    heads_num = 4
    d_ff = 256
    N = 3
    dropout_rate = 0.1
    layer_norm_eps = 1e-8
    pad_idx = 0
    batch_size = 100
    lr = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epoch = 10

    """
    3. define model
    """
    net = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len,
        d_model=d_model,
        heads_num=heads_num,
        d_ff=d_ff,
        N=N,
        dropout_rate=dropout_rate,
        layer_norm_eps=layer_norm_eps,
        pad_idx=pad_idx,
        device=device,
    )

    """
    4. define dataset and dataloader
    """
    def src_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
        return text_to_tensor(text, src_vocab, max_len, eos=False, bos=False)

    def src_tensor_to_text(tensor: torch.Tensor) -> str:
        return tensor_to_text(tensor, src_vocab)

    def tgt_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
        return text_to_tensor(text, tgt_vocab, max_len)

    def tgt_tensor_to_text(tensor: torch.Tensor) -> str:
        return tensor_to_text(tensor, tgt_vocab)

    train_dataset = KfttDataset(
        TRAIN_SRC_CORPUS_PATH,
        TRAIN_TGT_CORPUS_PATH,
        max_len,
        src_text_to_tensor,
        tgt_text_to_tensor,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = KfttDataset(
        VAL_SRC_CORPUS_PATH,
        VAL_TGT_CORPUS_PATH,
        max_len,
        src_text_to_tensor,
        tgt_text_to_tensor,
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = KfttDataset(
        TEST_SRC_CORPUS_PATH,
        TEST_TGT_CORPUS_PATH,
        max_len,
        src_text_to_tensor,
        tgt_text_to_tensor,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    """
    5. train
    """
    trainer = Trainer(
        net,
        optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True), # TODO: amsgradについて調べる
        nn.CrossEntropyLoss(ignore_index=pad_idx),
        BleuScore(tgt_vocab),
        device,
    )
    train_losses: list[float] = []
    train_bleu_scores: list[float] = []
    val_losses: list[float] = []
    val_bleu_scores: list[float] = []

    for i in range(max_epoch):
        print(f"epoch: {i + 1} \n")
        (
            avg_train_loss,
            avg_bleu_score,
            avg_val_loss,
            avg_val_bleu,
        ) = trainer.fit(train_loader, val_loader, print_log=True, epoch=i + 1, max_epoch=max_epoch)

        train_losses.append(avg_train_loss)
        train_bleu_scores.append(avg_bleu_score)
        val_losses.append(avg_val_loss)
        val_bleu_scores.append(avg_val_bleu)
        torch.save(trainer.net, join(NN_MODEL_PICKLES_PATH, f"epoch_{i}.pth"))


    """
    6. Test
    """
    test_losses, test_bleu_scores = trainer.test(test_loader)
    print(f"test loss: {sum(test_losses) / len(test_losses):.4f}")
    print(f"test BLEU: {sum(test_bleu_scores) / len(test_bleu_scores):.4f}")

    """
    7. plot and save
    """
    fig = plt.figure(figsize=(24, 8))

    # 左側: loss
    loss_ax = fig.add_subplot(1, 2, 1)

    # 右側: BLEU
    bleu_ax = fig.add_subplot(1, 2, 2)

    # epoch番号。1, 2, 3, ... の形にする
    epochs = list(range(1, max_epoch + 1))

    # lossを描画
    loss_ax.plot(epochs, train_losses, label="train loss")
    loss_ax.plot(epochs, val_losses, label="val loss")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    # BLEUを描画
    bleu_ax.plot(epochs, train_bleu_scores, label="train BLEU")
    bleu_ax.plot(epochs, val_bleu_scores, label="val BLEU")
    bleu_ax.set_title("BLEU Score")
    bleu_ax.set_xlabel("Epoch")
    bleu_ax.set_ylabel("BLEU")
    bleu_ax.legend()

    # レイアウトを整える
    fig.tight_layout()

    # 保存
    plt.savefig(join(FIGURE_PATH, "loss_bleu.png"))
    plt.close()
