# M2

一個原生 DeepSeek V4 HCA＋CSA decoder block（兩個 layers），啟用 scan；壓縮比例 `[128, 4]`，沒有額外 prefix layers。V=65,536、D=2,048；8 個 routed experts、top-2，加 1 個 shared expert，各寬度 2,048，每 token 啟用寬度共 6,144。mHC expansion=4，16 個 query heads、1 個共享 KV head、head_dim=128，Q LoRA rank=512。

Muon、Tokamax、SEQ_MINOR、fused MLP、save_qkv_proj remat。DeepSeek V4 使用原生 compressed Q/KV projections，因此關閉 Qwen3 的 fused QKV。BF16 權重與主要矩陣運算；logits、loss、梯度累積及 momentum 使用 FP32，沒有 FP32 master weights。`model_name=deepseek4-tiny` 選用 MaxText 的 V4 routing 路徑，`override_model_config=true` 以本專案 JSON 覆寫尺寸，不會建立官方 tiny preset 的 43 層模型。

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

程式沿用環境變數、不印出 token。直接讀取 `configs/curriculum.json`，不修改 microbatch 或 GA，不執行任何測速。每次更新容量 512K tokens；microbatch 與 GA 依你的 curriculum 設定。microbatch 是全域 sequence 數，每裝置 batch 依裝置數換算。

資料由 HF datasets 即時 streaming，讀取小批文件後執行 BPE，直接送入訓練；不先準備整份 10B tokens、不寫入本地訓練 Parquet、不 shuffle。訓練進度條顯示該長度階段的 step、loss、速度與預估剩餘時間，續訓會跳到已完成的 step。首次讀取／編譯期間，尚未完成的 step 不會提前計入。

關閉 HF／datasets 自帶的 `Resolving data files` 等進度條。只保留固定位置動態更新的訓練進度條，完成後不保留舊進度條。

Tokenizer 使用同程序 Rust 多執行緒的 `encode_batch`，預設每批 256 篇文件、最多 8 個 threads，不建立 Python worker 程序。可調整：

```python
%run -m llm5.start --tokenizer-batch-size 256 --tokenizer-threads 8
```

若已自行設定 `RAYON_NUM_THREADS`，會沿用該值。Rust thread pool 已啟動後，修改 thread 數需要在下一次 kernel 啟動前設定環境變數。

已訓練的 tokenizer 在 `models/tokenizer`。Streaming 僅讀取 `content` 欄位，關閉 Arrow 整檔預讀；下載量仍由遠端 Parquet 區塊決定，不等於文字或 token 大小。記憶體包含 Arrow 解碼區塊、tokenizer batch 與各來源的 token buffer，沒有整份資料集的常駐副本。

Checkpoint 每 **100 個 optimizer step** 儲存一次，階段結束時也會儲存。完整 Orbax checkpoint（模型、optimizer state、HF streaming cursor、抽樣 RNG 與剩餘 token buffer）由同一程序內的背景執行緒上傳至 [tim123xyz/Test](https://huggingface.co/tim123xyz/Test)，沿用 `HF_TOKEN`，需要該模型庫的寫入權限。全程不啟動另一個訓練程序。

本機預設在 `/content/M2/runs/checkpoints`，每階段保留最近 3 份；待上傳資料使用快照保護，不會被本機保留策略提前移除。只有完整上傳成功後才更新 Hub 的 `latest.json`。階段結束和正常離開程式前會等候剩餘上傳完成；背景錯誤會在下一步直接拋出。Hub 保留每份上傳的 checkpoint，不自動刪除遠端版本。

TensorBoard logdir 固定為 `<output-root>/<run-name>/tensorboard`。每個 checkpoint 先 flush writer、複製 events 快照，再由背景執行緒上傳至 Hub 的 `tensorboard/<run-name>`；階段結束後另存最後一批 events。續訓會從已提交的 revision 恢復所有階段的 logs 到原路徑，再建立新 event file 接續記錄，不需手動下載。事件檔採複製而非 hard link，確保背景上傳時內容不會繼續變動。異常中斷最多遺失最後一次同步後的 logs。

可在另一個 Colab cell 提前開啟即時監視：

```python
%load_ext tensorboard
%tensorboard --logdir /content/M2/runs/checkpoints
```

DeepSeek V4 預設 run prefix 是 `m2-deepseek4-1block-512k-v1`；Hub 在 `runs/<run-prefix>/latest.json` 保存獨立續訓索引，同時更新根目錄 `latest.json`。新架構從頭訓練，保留舊 Qwen3 的 checkpoint；兩者權重不相容。

下次執行相同命令，會自動讀取 Hub 最新 checkpoint，驗證 model／curriculum／tokenizer／輸入格式是否匹配，下載並恢復完整訓練狀態；若本機有更新的 checkpoint 則優先使用本機。已完成階段不會重新訓練；切換長度階段沿用串流 cursor 與未使用的 tokens。新 Colab runtime 直接恢復 streaming，不需要重建訓練資料。舊版預先打包資料的 iterator 不相容，會直接報錯，避免默默重置資料位置。這些是 MaxText 訓練 checkpoint，並非直接供 `AutoModel.from_pretrained` 使用的 Transformers 權重。

預設即上傳並自動續訓；也可以明確指定：

```python
%run -m llm5.start --hub-repo tim123xyz/Test
```

所有訓練在目前 kernel 執行，原始錯誤直接拋出。JSON 設定在 `configs`；MaxText 使用 PyPI 版本 0.2.5，不 clone、不固定 commit、不建立 venv、不限制 TPU 型號。

Notebook：[M2_TPU_v6e_512K.ipynb](notebooks/M2_TPU_v6e_512K.ipynb)。
