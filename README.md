# M2

五層 Qwen3 decoder，V=65,536、D=2,048、Dff=6,144。Muon、Tokamax、SEQ_MINOR QKV、fused QKV/MLP、save_qkv_proj remat。BF16 權重與主要矩陣運算；logits、loss、梯度累積及 momentum 使用 FP32，沒有 FP32 master weights。

## Colab

首次取得專案並安裝：

```python
!git clone https://github.com/GarD0m/M2.git /content/M2
%cd /content/M2
%pip install -r requirements.txt
```

先執行你的環境設定：

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/jax-cache"
```

開始／續訓：

```python
%cd /content/M2
%run -m llm5.start
```

程式沿用環境變數、不印出 token。直接讀取 `configs/curriculum.json`，不修改 microbatch 或 GA，不執行任何測速。依目前設定：microbatch 為 128／64／32／16／8／4，GA 都為 4，每次更新容量 512K tokens。microbatch 是全域 sequence 數；每裝置 batch 依裝置數換算，但不自動改寫 curriculum。

缺少資料時先從 UltraData L3 準備約 10B tokens，int32 原始容量約 40GB，另需 cache 與 checkpoint 空間；資料已完成則直接驗證並訓練。使用 HF datasets 串流、不 shuffle，已訓練的 tokenizer 在 `models/tokenizer`。下載量由 Parquet 區塊決定，不等於文字樣本大小。

Checkpoint 預設在 `/content/M2/runs/checkpoints`；需保存至 GCS 時：

```python
%run -m llm5.start --output-root gs://YOUR_BUCKET/m2
```

所有訓練在目前 kernel 執行，原始錯誤直接拋出。JSON 設定在 `configs`；MaxText 使用 PyPI 版本 0.2.5，不 clone、不固定 commit、不建立 venv、不限制 TPU 型號。

Notebook：[M2_TPU_v6e_512K.ipynb](notebooks/M2_TPU_v6e_512K.ipynb)。
