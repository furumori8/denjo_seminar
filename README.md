# CIFAR-10 ResNet モデル仕様

## モデルアーキテクチャ
- `Conv2d(3→32)` → BatchNorm → ReLU で入力画像の初期特徴抽出。
- ResidualBlock（3×3 畳み込み×2、BatchNorm、ReLU、Dropout）を計 4 層スタックし、チャンネル数を 32→64→128→256→512 と拡張。
- ストライド変更やチャンネル数が変わる箇所は 1×1 畳み込み + BatchNorm でショートカットを調整。
- `AdaptiveAvgPool2d(1,1)` → Flatten → Dropout(0.1) → `Linear(512→10)` でクラスロジットを出力。
- 損失関数には `LabelSmoothingLoss(classes=10, smoothing=0.1)` を使用。

## データ前処理
- データセット: `torchvision.datasets.CIFAR10(train=True/False, root='~/tmp/cifar')`
- 訓練時の変換: RandomHorizontalFlip、RandomCrop(32, padding=4)、RandomErasing(p=0.5)、Normalize(mean=[0.5071,0.4867,0.4408], std=[0.2675,0.2565,0.2761])
- 検証／テスト時: Normalize のみ
- バッチサイズ: 訓練・検証 64、テスト 1

## 学習フロー
- KFold(n_splits=5, shuffle=True, random_state=44) で訓練データを 5 分割。
- 各 Fold ごとに `CIFAR10ResNet(num_classes=10, dropout_prob=0.1)` を初期化し GPU へ転送。
- Optimizer: Adam(lr=1e-3)  
  Scheduler: CosineAnnealingLR(T_max=40, eta_min=1e-5)
- 自動混合精度: `torch.amp.autocast(device_type='cuda')` + `GradScaler`
- 学習エポック数: 40。ループ内で損失と精度をログ出力。

## 検証処理
- `val` 関数で `Net.eval()` / `torch.no_grad()` を用い、各ミニバッチのロジットから損失と精度を積算。
- Fold ごとに平均損失・精度を計算し、標準出力に表示。

## 推論とアンサンブル
- テストデータを DataLoader(batch_size=1) で処理。
- 学習済み 5 モデルのロジットを平均化し、`torch.max` で最終クラスを決定。
- 予測結果を `submition.csv`（image_id, labels）として保存。

## 出力・ログ
- `submition.csv`: テストデータの予測ラベル。
- 標準出力: 各 Fold の学習経過、検証損失・精度、Fold 完了メッセージ。
