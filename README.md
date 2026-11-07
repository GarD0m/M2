# M2

一個 DeepSeek V4 HCA＋CSA decoder block（兩個 layers），使用原生 scan 跑一輪。V=65,536、D=2,048。每層有 16 個 routed experts，每 token 選 3 個，每個 expert 寬度 2,048。總參數約 5.46 億，精確值由 MaxText 在初始化時統計。

使用 Muon、Tokamax Splash（SEQ_MINOR）與 Tokamax GMM v2。DeepSeek V4 使用原生 compressed Q/KV projections 和 routed MoE。BF16 權重與主要矩陣運算，logits、loss、梯度累積及 momentum 使用 FP32。`model_name=deepseek4-tiny` 選用 MaxText 的 V4 routing 路徑，`override_model_config=true` 以本專案 JSON 覆寫尺寸。

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

自訂 remat 在 TPU 上保留 Q/KV projections 和 MoE top-k 路由索引。MoE 反向重用排序索引，router 梯度採用 MaxText 的 dense VJP。模型參數與 optimizer state 也留在 TPU。單裝置下，CSA teacher 在 1K／2K 分別以 4／2 heads 為一組，4K 起每組 1 head。訓練的 logits 最多分成 16 塊；單裝置每 microbatch 32768 tokens 時，每塊為 2048 tokens，單一 FP32 logits 張量為 512 MiB。實際分塊數依 token 數對齊，驗證依 batch 縮減分塊數，維持 128 tokens 對齊。mHC RMSNorm 和 HyperHead 的大型運算元使用 BF16，歸約保留 FP32。

CSA indexer 每個 token 最多選取 64 個壓縮區塊，減少 trace 中耗時的 mask gather；這比原先的 128 更稀疏，會改變注意力與訓練結果。實際吞吐、MFU 和品質影響需在 TPU 上驗證。

mHC 的 Sinkhorn 使用 10 次迭代，保留 FP32 正規化。

開始／續訓：

```python
%cd /content/M2
%run -m llm5.start
```

程式沿用環境變數、不印出 token。直接讀取 `configs/curriculum.json`，不修改 microbatch 或 GA，不執行任何測速。每次更新容量 512K tokens；microbatch 與 GA 依你的 curriculum 設定。microbatch 是全域 sequence 數，每裝置 batch 依裝置數換算。

資料由 HF datasets 即時 streaming，讀取小批文件後執行 BPE，直接送入訓練；不先準備整份 10B tokens、不寫入本地訓練 Parquet、不 shuffle。訓練進度條顯示該長度階段的 step、loss、速度與預估剩餘時間，續訓會跳到已完成的 step。首次讀取／編譯期間，尚未完成的 step 不會提前計入。

關閉 HF／datasets 自帶的 `Resolving data files` 等進度條。只保留固定位置動態更新的訓練進度條，完成後不保留舊進度條。

Tokenizer 使用同程序 Rust 多執行緒的 `encode_batch_fast`，略過訓練用不到的字元位置。分詞結果在主機端用 NumPy 批次組裝。預設每批 256 篇文件、最多 8 個 threads。可調整：

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

TensorBoard 只寫入訓練 loss、整次驗證的平均 loss、學習率和每裝置 tokens/s。

## JAX 瓶頸分析

在 Colab 執行獨立的 MaxText JAX XPlane 測試：

```python
%run -m llm5.profile --length 1024 --microbatch 32 --ga 16 --warmup-steps 1 --profile-steps 4
```

測試只執行一組由 CLI 指定的配置，不讀取 curriculum；`microbatch` 是全域 sequence 數，須能被 TPU 裝置數整除。預設長度 1024、microbatch 32、GA 16；可用 `--math` 和 `--code` 指定資料比例，預設為 0.05 和 0.15。模型和資料來源仍沿用訓練設定。`--warmup-steps` 指定擷取前跳過的 optimizer steps（預設 1，可設 0），`--profile-steps` 指定擷取步數（預設 4，至少 1）；總共執行兩者相加再多 1 步，以完成 trace。每次測試使用新的本機目錄，不讀取或儲存訓練 checkpoint，也不上傳 Hub。正式訓練仍使用 `llm5.start`。

使用相同的 `--length`、`--microbatch` 和 `--ga`，分別執行 `--vocab-tiles 16` 與 `--vocab-tiles 32` 才能比較分塊的影響。16 塊的 logits 張量可能需要兩倍記憶體。可用 `--indexer-topk 64` 和 `--indexer-topk 128` 比較 CSA 稀疏度；可用 `--csa-head-chunk` 測試 CSA teacher 的分塊大小，預設沿用訓練計畫的自動設定。每次 profiler 都會將配置寫入測試目錄的 `profile-config.json`，每次執行都需使用新的 runtime。

可加入 `--mhc-lite`，在獨立 profiler 測試 MaxText 的 mHC-lite TPU Pallas 融合算子，並與不加此參數的 trace 比較。mHC-lite 使用不同的殘差映射和參數形狀，不能直接沿用目前的訓練 checkpoint；此選項目前只用於 profiler，尚未確認速度或記憶體效果。

程式會印出本次 TensorBoard logdir，位於 `runs/profiles/<測試識別碼>/tensorboard`；XProf trace 位於該目錄的 `plugins/profile/`，編譯時的 HLO 文字與 protobuf dump 則位於同一測試目錄的 `hlo/`。在 Colab 安裝 `xprof`（`%pip install xprof`），以該 logdir 開啟 XProf 的 **Trace Viewer** 等工具，分析主機、TPU kernel 與運算時間；HLO 檔案可另用於檢查編譯結果。測試仍會執行 loss 與梯度計算，但不將訓練指標寫入 TensorBoard。檔案留在 Colab runtime 本機，HLO dump 可能佔用大量空間。

HLO dump 的 XLA 設定必須在 JAX 後端初始化前生效。請在新的 Colab runtime 執行 profiler；每次重新執行 profiler 或切換回正式訓練前，也請重新啟動 runtime。Profiler 會停用該程序的 JAX 編譯快取以確保重新編譯，並保留原有的 `XLA_FLAGS`；若其中已設定 `--xla_dump_to`，需先移除該項。

DeepSeek V4 預設 run prefix 是 `m2-deepseek4-1pass-16e-512k-v1`。Hub 在 `runs/<run-prefix>/latest.json` 保存獨立續訓索引，同時更新根目錄 `latest.json`。單輪訓練使用自己的 checkpoint 索引。

下次執行相同命令，會自動讀取 Hub 最新 checkpoint，驗證 model／curriculum／tokenizer／輸入格式是否匹配，下載並恢復完整訓練狀態；若本機有更新的 checkpoint 則優先使用本機。已完成階段不會重新訓練；切換長度階段沿用串流 cursor 與未使用的 tokens。新 Colab runtime 直接恢復 streaming，不需要重建訓練資料。舊版預先打包資料的 iterator 不相容，會直接報錯，避免默默重置資料位置。這些是 MaxText 訓練 checkpoint，並非直接供 `AutoModel.from_pretrained` 使用的 Transformers 權重。

預設即上傳並自動續訓；也可以明確指定：

```python
%run -m llm5.start --hub-repo tim123xyz/Test
```

所有訓練在目前 kernel 執行，原始錯誤直接拋出。JSON 設定在 `configs`；MaxText 使用 PyPI 版本 0.2.5，不 clone、不固定 commit、不建立 venv、不限制 TPU 型號。

Notebook：[M2_TPU_v6e_512K.ipynb](notebooks/M2_TPU_v6e_512K.ipynb)。
