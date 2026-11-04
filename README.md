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

資料準備顯示來源連線、階段、寫入 token 數、已處理文件數及 shard 進度；訓練顯示該長度階段的 step、loss、速度與預估剩餘時間，續訓會跳到已完成的 step。下載／編譯期間尚未完成的 token 或 step 不會提前計入。

Tokenizer 使用同程序 Rust 多執行緒的 `encode_batch`，預設每批 256 篇文件、最多 8 個 threads，不建立 Python worker 程序。可調整：

```python
%run -m llm5.start --tokenizer-batch-size 256 --tokenizer-threads 8
```

若已自行設定 `RAYON_NUM_THREADS`，會沿用該值。Rust thread pool 已啟動後，修改 thread 數需要在下一次 kernel 啟動前設定環境變數。

缺少資料時先從 UltraData L3 準備約 10B tokens，int32 原始容量約 40GB，另需 cache 與 checkpoint 空間；資料已完成則直接驗證並訓練。使用 HF datasets 串流、不 shuffle，已訓練的 tokenizer 在 `models/tokenizer`。下載量由 Parquet 區塊決定，不等於文字樣本大小。

Checkpoint 每 **100 個 optimizer step** 儲存一次，階段結束時也會儲存。完整 Orbax checkpoint（模型、optimizer state、Grain iterator）由同一程序內的背景執行緒上傳至 [tim123xyz/Test](https://huggingface.co/tim123xyz/Test)，沿用 `HF_TOKEN`，需要該模型庫的寫入權限。全程不啟動另一個訓練程序。

本機預設在 `/content/M2/runs/checkpoints`，每階段保留最近 3 份；待上傳資料使用快照保護，不會被本機保留策略提前移除。只有完整上傳成功後才更新 Hub 的 `latest.json`。階段結束和正常離開程式前會等候剩餘上傳完成；背景錯誤會在下一步直接拋出。Hub 保留每份上傳的 checkpoint，不自動刪除遠端版本。

下次執行相同命令，會自動讀取 Hub 最新 checkpoint，驗證 model／curriculum／tokenizer 是否匹配，下載並恢復完整訓練狀態；若本機有更新的 checkpoint 則優先使用本機。已完成階段不會重新訓練。若新 Colab runtime 沒有原訓練資料，仍需重新準備相同資料，之後才按保存的 iterator 位置續訓。這些是 MaxText 訓練 checkpoint，並非直接供 `AutoModel.from_pretrained` 使用的 Transformers 權重。

預設即上傳並自動續訓；也可以明確指定：

```python
%run -m llm5.start --hub-repo tim123xyz/Test
```

所有訓練在目前 kernel 執行，原始錯誤直接拋出。JSON 設定在 `configs`；MaxText 使用 PyPI 版本 0.2.5，不 clone、不固定 commit、不建立 venv、不限制 TPU 型號。

Notebook：[M2_TPU_v6e_512K.ipynb](notebooks/M2_TPU_v6e_512K.ipynb)。
