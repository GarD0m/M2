# LLM5：Qwen3 decoder 在單顆 TPU v6e 預訓練

本專案從隨機權重訓練五層 decoder，使用 Google MaxText／JAX／Flax／Optax／Grain。沒有使用 Qwen3 預訓練權重。依目前指示，**不建立雲端資源，也不啟動雲端訓練**。

## 模型與精度

| 項目 | 設定 |
|---|---|
| Decoder | Qwen3 dense，5 layers |
| V / D / Dff | 65,536 / 2,048 / 6,144 |
| Attention | GQA，16 query heads／8 KV heads，head dim 128 |
| Block | QK RMSNorm、RMSNorm、SwiGLU、RoPE theta 1,000,000 |
| Embedding | input/output tied；估算 **385,899,776 parameters** |
| Optimizer | 原生 Optax Muon；embedding、norm 等使用其 AdamW 分支 |
| 精度 | BF16 weights/compute/activation/momentum；不保留 FP32 master weights；梯度累積與 loss 保留 FP32 |
| Remat | `save_qkv_proj`：保留 Q/K/V 投影，其餘 activation 重算 |
| Attention kernel | Tokamax Splash；Q/K/V 均用 SEQ_MINOR；fused backward、reciprocal、base2 exp |
| Fusion | fused QKV projection、fused gate/up MLP projection |
| Loss | `num_vocab_tiling=32`，分塊計算 64K vocab 的 cross entropy |
| LR | WSO = **Warmup → Stable only**；64 optimizer steps warmup（約 33.6M tokens），之後維持 0.001 |

K 按 1024 解讀；10B token budget 按十進位。GQA head 數、tied embedding、LR 與 warmup 是可修改的起始選擇，並非已由實驗找到的最佳值。BF16 momentum 使用原生 `mu_dtype`；數值敏感部分保留 FP32。

## Sequence length curriculum

常用名稱為 **sequence length curriculum（序列長度課程）**，設定在 [`configs/curriculum.json`](configs/curriculum.json)。比例是 math/code 所屬資料來源的 token 比例，不是難度等級。

| Context | 約 token 預算 | Math | Code | General | 初始 microbatch | GA | Optimizer steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2B | 5% | 15% | 80% | 32 | 16 | 3,818 |
| 2K | 2B | 7% | 17% | 76% | 16 | 16 | 3,817 |
| 4K | 2B | 9% | 19% | 72% | 8 | 16 | 3,816 |
| 8K | 2B | 11% | 21% | 68% | 4 | 16 | 3,815 |
| 16K | 1B | 13% | 23% | 64% | 2 | 16 | 1,907 |
| 32K | 1B | 15% | 25% | 60% | 1 | 16 | 1,907 |

單顆 TPU 上 `per_device_batch_size` 是 microbatch；GA 為 `gradient_accumulation_steps`。每次 optimizer update 的容量為 **524,288 tokens（512K）**，總計約 **10.0B loss tokens／19,080 updates**。有效 sequence batch 依序為 512、256、128、64、32、16。以完整 batch 取整，不做精確 token 尾批截斷。Native Grain shift 會遮罩每列最後一個 target，預算估算已扣掉此位置。

完整 notebook：[M2_TPU_v6e_512K.ipynb](notebooks/M2_TPU_v6e_512K.ipynb)；各 code block 的文字版：[BLOCKS.md](notebooks/BLOCKS.md)。原本的安裝、路徑設定、tokenizer 驗證及 plan 產生四格（舊第 2–5 格）已移除。安裝獨立執行 `%pip install -r requirements.txt`；notebook 現在依序為取得專案、tune、準備資料、驗證、訓練。Tune 的 `--prepare-plan` 自動以預設路徑產生 plan，完成後套用測速選出的 microbatch。不建立 venv、不限制或檢查 TPU 型號；只讀取裝置數換算 batch，確保有效 batch 仍為 512K tokens。已完成的 tokenizer 隨專案放在 `models/tokenizer`。

每階段使用獨立 `run_name`。`steps` 是累計結束步數；後一階段從前一階段的 **full state checkpoint** 載入，保留 weights、Muon/AdamW momentum、optimizer counters 和全域 step。新的長度建立新的資料 iterator；同一階段中斷續訓則由 MaxText 自動載入該階段 checkpoint 和 Grain iterator。六階段共用完整 LR schedule，不重新 warmup、不在最後衰減。

## 本機檢查

Python 3.12+；準備資料與 tokenizer 不需要 TPU。

```powershell
python -m pip install -e '.[test]'
python -m pytest -q
python -m llm5.plan
python -m llm5.launch
```

最後一個指令預設只列印命令。`runs/plan` 是本機產生的預覽，搬到 Linux TPU VM 後要重新產生，不能沿用 Windows 絕對路徑。

## TPU VM 上的流程

Colab 使用上方 notebook 的 `%pip install -r requirements.txt`。使用 PyPI 版本 `maxtext==0.2.5`，不 clone MaxText、不固定 commit。一般 Linux TPU 環境也可以用以下腳本，直接安裝至目前 Python，不建立 venv：

```bash
bash scripts/setup_tpu.sh

# 使用 Hugging Face datasets streaming，只讀 content，小批次取樣。
# 字元上限控制 BPE 文字樣本量，不是網路流量上限。
python -m llm5.tokenizer

# 原生 Rust tokenizers encoding → 等長 token blocks → Parquet。
# 10B int32 tokens 原始大小約 40GB；另需預留 cache、checkpoint 與暫存空間。
python -m llm5.data

# output-root 可為本機絕對路徑，正式訓練建議使用自己的 GCS prefix。
python -m llm5.plan --output-root gs://YOUR_BUCKET/llm5 --run-prefix experiment-001
python -m llm5.launch --check-data

# 在 TPU 做短程 synthetic training，檢查完整模型和 GA 的記憶體。
python -m llm5.tune
python -m llm5.plan --output-root gs://YOUR_BUCKET/llm5 --run-prefix experiment-001 \
  --microbatches runs/microbatches.json

# 實際啟動；只在準備好執行完整預訓練時使用。
python -m llm5.launch --execute
```

`tune` 在目前程序直接呼叫 MaxText，依序測各階段的 2 次方 microbatch 候選，不啟動訓練子程序。GA 隨之調整，維持每次更新 512K tokens，預設搜尋每裝置 microbatch 容量不超過 128K tokens 的候選。每個候選測 12 步，捨棄前 4 步，依後續 step time 中位數選最快的候選。成功後清理 JAX 編譯快取及可回收物件，再測下一個候選。所有錯誤（包括 OOM）直接顯示原始 traceback 並停止，不包裝或跳過。結果輸出 `runs/microbatches.json` 和完整 timing report。**目前未在 TPU 上執行，因此不保證起始 microbatch 已能放入記憶體，也沒有訓練速度或品質結果。** Google 的 v6e 文件列出每顆 32GB HBM：[官方規格](https://docs.cloud.google.com/tpu/docs/v6e)。

以起始 microbatch 的 32K tokens 計算：BF16 權重約 0.719 GiB，比 FP32 權重減少約 0.719 GiB；FP32 gradient buffer 約 1.438 GiB；五層 QKV checkpoint 約 1.25 GiB；layer inputs 約 0.625 GiB；單個 FP32 logits tile 約 0.25 GiB。這些是個別張量大小，**不是總 HBM 預測**：optimizer states、Muon Newton–Schulz workspace、重算 activation、XLA 暫存與存活時間仍需實機 profiling。每階段的計算結果包含在 plan manifest。

訓練入口 `llm5.train` 在原生 MaxText 啟動前補上 Muon 的 `qkv_proj` 維度辨識，將融合矩陣 `[D, Hq+2Hkv, head_dim]` 視為 `[D, (Hq+2Hkv)*head_dim]`。fused gate/up 的 activation 軸則保留為兩組獨立 Muon 矩陣。融合 QKV 的 Muon 更新作用於聯合矩陣，並非三個獨立 Q/K/V 更新的數值等價替換；切換 fusion 或 weight dtype 後使用新的訓練 run，不直接沿用舊 optimizer checkpoint。實驗用 Splash scheduler 維持關閉。

## 資料與 tokenizer

[`configs/sources.json`](configs/sources.json) 固定三組資料來源的 revision 與子集：

- General：[Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3)，中英文、QA 與 multi-style 四個子集。
- Math：[UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math)，只取四個 L3 子集。
- Code：[UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code)，只取 `UltraData-Code-L3` 的 11 種語言 splits。

各子集使用已查核的 `content` 欄位；Code L3 content 包含題目與解答，不把 `raw_content` 額外重複拼接。General 內四個子集、Math 內四個子集、Code 內十一個語言，目前採等 document 權重。中英文的實際 token 比例不保證各半，可在 sources 設定調整。

訓練 BPE 時以 70/10/20 的字元配額選取 General/Math/Code，預設最多 20M characters。使用原生 `datasets` 串流，**不呼叫 shuffle**，依序只開啟一個子集 reader，先將樣本存成本機 `*-hf-sample.jsonl`，關閉資料 reader 後才離線訓練 BPE。字元配額不是下載 byte 上限：Parquet 仍需讀取壓縮 column chunk，單個 chunk 可超過 100MB。中斷的 `.partial` 樣本不會當作完成樣本；已完成樣本可用 `--sample-file 路徑` 離線重用。

數字逐字切分、保留 code 空白與換行、不做 Unicode normalization，初始完整 byte alphabet 可 round-trip UTF-8。詞彙不足 64K 會要求更多樣本，不用假 reserved tokens 補足。產物包含 Hugging Face `tokenizer.json`／設定，供 MaxText 原生 Hugging Face tokenizer 讀取。Chat template 只是格式定義；此預訓練沒有使模型成為 instruction/chat model。

資料預處理依內容 SHA-256 留 0.5% validation，其餘 train；tokenizer 僅讀 train partition。同 domain 的文件以 EOS 串接，切成相同長度的 block，再按 curriculum 配額混合，所以 math/code 比例不會因文件長度不同而偏移。跨階段沿用 token stream，資料耗盡會報錯，不默默循環。Validation 使用固定 70/10/20 domain block 比例。

這裡採 continuous packing：同 block 內 EOS 兩側可互相 attention，沒有 document boundary isolation。32K 階段能含多篇短文件；這提供長度訓練曝光，並不等於已有原生 32K 長文推理能力。需要另外評估長文檢索與長距離依賴。

每階段資料完成後寫入 metadata，記錄 tokenizer／sources／curriculum hashes 與 sequence 數。啟動時檢查 hash、Parquet rows 和抽樣 token 範圍，防止換 tokenizer 後誤用舊資料。預處理輸出必須是新目錄；中斷的未完成資料請改用新目錄重建，沒有宣稱具備 preprocessing resume。

`requirements.txt` 是專案維護的 Colab TPU 安裝清單，參照使用者提供的 `pip-freeze.tpu.txt`，保留 JAX/JAXlib 0.11.2、libtpu 0.0.50 等核心版本，並使用符合 MaxText NNX 需求的 Flax。原始 freeze 只作參考，不直接重裝 Colab 本機套件路徑。MaxText 不使用 `[tpu]` extra，因為其 0.2.5 套件宣告包含 SymPy/mpmath 衝突；本專案明列 runtime 依賴，採用相容的 SymPy 1.14.0／mpmath 1.3.0。已完成 Linux Python 3.13 的依賴解析，尚未在 Colab TPU 執行完整安裝及訓練。

專案設定全部使用 JSON：`configs/model.json`、`configs/curriculum.json`、`configs/sources.json`、`configs/maxtext-base.json`，以及產生的 stage 和 probe 設定。MaxText 0.2.5 預設設定已轉成專案內的 `maxtext-base.json`；不修改安裝的套件。`llm5.train` 補上 `.json` 設定入口。每個 probe 使用獨立 JSON 文件，明確設定 `"load_full_state_path": ""` 和 `"load_parameters_path": ""`，不透過 CLI 的空值覆寫，以免被解析成 null。Notebook 用 `%run` 在同一 kernel 執行，裝置探測與訓練共用 TPU runtime。單 host Colab 的設定採 `skip_jax_distributed_system=true`、同步 checkpoint，避免 kernel 已持有 TPU 時再啟動另一個 runtime。

寫入 TensorBoard 前檢查指標中的 NaN/Inf，並顯示 step、phase、完整指標名稱與異常值。Loss、梯度 norm 或參數 norm 異常會直接拋出 `FloatingPointError`；其他輔助指標發出具名 warning，不修改或隱藏原始數值。TensorBoard 的泛用 NaN/Inf warning 本身不足以判斷 loss 是否發散。

安裝 requirements 並進入專案目錄後，直接執行 synthetic batch/GA 測速，不需要先執行其他設定 blocks 或準備訓練資料：

```bash
%run -m llm5.tune --prepare-plan --manifest runs/plan-512k/manifest.json --out runs/microbatches-512k.json --log-dir runs/probes-512k --steps 12 --skip-steps 4 --max-micro-tokens 131072
```

官方框架參考：[MaxText 0.2.5](https://pypi.org/project/maxtext/0.2.5/)、[安裝流程](https://maxtext.readthedocs.io/en/latest/install_maxtext.html)。
