# transformer-scratch

PyTorchを用いたTransformerのスクラッチ実装レポジトリです。

参考文献
- https://zenn.dev/yukiyada/articles/59f3b820c52571
- https://arxiv.org/pdf/1706.03762

## セットアップ
1. `poetry install` を実行する


## プロジェクト構成
。
├── const // pathなどの定数値
│   └── path.py
├── corpus // 訓練用のデータ・コーパスが入る
│   └── kftt-data-1.0
├── figure
├── layers // 深層ニューラルネットを構成するレイヤの実装
│   └── transformer
│       ├── Embedding.py
│       ├── FFN.py
│       ├── MultiHeadAttention.py
│       ├── PositionalEncoding.py
│       ├── ScaledDotProductAttention.py
│       ├── TransformerDecoder.py
│       └── TransformerEncoder.py
├── models // 深層ニューラルネットモデルの実装
│   ├── Transformer.py
│   └── __init__.py
├── mypy.ini
├── pickles // モデルやデータセットのpickleファイルを格納
│   └── nn/
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── tests // テスト(pytest)
│   ├── conftest.py
│   ├── layers/
│   ├── models/
│   └── utils/
├── train.py // 訓練用コード
└── utils // DatasetやVocabといったクラスの実装,前処理に用いる関数の実装
    ├── dataset/
    ├── download.py
    ├── evaluation/
    └── text/
