# PrefillBench：Prefill 吞吐与延迟测试

使用真实对话和 agent 轨迹，测试 **Qwen3-8B-FP8** 在不同请求长度、并发和 KV cache 条件下的性能。

使用 **SGLang 0.5.19**，运行入口为 [`run_serve_bench.sh`](run_serve_bench.sh)，配置为 [`shape.yaml`](shape.yaml)。测试经过真实调度器，测量不同并发下的 prefill 吞吐、请求延迟和 KV cache 复用效果。

| 角色 | GPU | 状态 |
|---|---|---|
| 参考基线 | 1 × H20（141 GB） | 已完成：[2026-10-15 报告](reports/h20_prefill_request_shapes_2026-10-15.pdf) |

## 快速开始

### 1. 准备机器和 Python 环境

- 使用 Python 3.11，先确认 NVIDIA 驱动和 CUDA 环境能够识别目标卡。
- 测试使用 SGLang 0.5.19。已有 H20 参考环境为 `torch 2.13.0+cu130`、驱动 `580.82.07`；目标机器的驱动需满足其 GPU 和 CUDA 环境的要求。
- 访问 GitHub、Python 包源和 Hugging Face。环境安装问题可参考 [SGLang 官方安装说明](https://docs.sglang.io/docs/get-started/install)，本测试固定使用 **0.5.19**。

执行以下命令：

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

git clone https://github.com/RebeccaZhang22/PrefillBench.git
cd PrefillBench

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "sglang==0.5.19" pyyaml pyarrow matplotlib huggingface_hub

python - <<'PY'
import sglang
import torch

print("SGLang:", sglang.__version__)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert sglang.__version__ == "0.5.19", "本测试使用 SGLang 0.5.19"
assert torch.cuda.is_available(), "当前 Python 环境无法使用 CUDA，请先检查驱动和安装环境"
print("CUDA 可用，GPU 数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    gpu = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {gpu.name}, 显存 {gpu.total_memory / 1024**3:.1f} GiB")
PY
```

先进入仓库并执行 `source .venv/bin/activate`。

### 2. 下载固定版本的数据和模型

在仓库根目录执行：

```bash
hf download allenai/WildChat-1M --repo-type dataset \
  --revision 7d6490e462285cf85d91eabea0f9a954fbddcd1f \
  --local-dir data/WildChat-1M

hf download nebius/SWE-rebench-openhands-trajectories --repo-type dataset \
  --revision 35455389ab51bf5e2306bfd436ef72d0f98bf882 \
  --local-dir data/SWE-rebench-openhands-trajectories

# 提前将模型下载到 Hugging Face 缓存；测试时会自动使用
hf download Qwen/Qwen3-8B-FP8
```

数据目录结构见 [`data/README.md`](data/README.md)。

### 3. 选择目标 GPU

打开 [`shape.yaml`](shape.yaml)，默认使用 GPU 0：

```yaml
runtime:
  python: python3
  sglang_version: "0.5.19"
  model: Qwen/Qwen3-8B-FP8
  gpus: [0]
  context_length: 81920
```

从 `nvidia-smi` 输出中找到目标 GPU。如果它是 GPU 3，只把 `gpus: [0]` 改为 `gpus: [3]`；只有一张卡时通常保持 `[0]`。这个入口只使用列表中的第一张卡，保持单卡测试即可。已激活 `.venv` 时，`python3` 就是该环境里的 Python。

其余参数先保持原样：`serve.latent_layer: 23` 表示只计算 block 0..23，共 24 层；保持数据、层数和上下文长度一致，便于比较不同 GPU 的结果。若数据或模型已放在其他目录，可修改 `sampling.sources[].input`、`runtime.model`；离线使用时还需将 `sampling.tokenizer` 指向同一模型目录。

### 4. 先做单并发试跑

下面的命令从正式配置生成 `shape_smoke.yaml`：四种请求形状都只测并发 1、prefill size 4096，缩短测量时间，先确认流程能走通。

```bash
python - <<'PY'
import yaml

with open("shape.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["serve"]["prefill_sizes"] = [4096]
cfg["serve"]["shapes"] = {name: [1] for name in cfg["serve"]["shapes"]}
cfg["serve"]["level_min_seconds"] = 10
cfg["serve"]["level_min_requests_per_conc"] = 1
cfg["serve"]["level_max_seconds"] = 60

with open("shape_smoke.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

bash run_serve_bench.sh shape_smoke.yaml
```

首次仍需对原始数据统计长度、抽样和分词，并加载模型、编译 kernel、预热。`level_max_seconds` 是每档停止发新请求的上限。

成功时，终端最后会打印 `done. report: .../serve_report.md`。正式测试会复用已经处理好的数据。

### 5. 运行完整基准

```bash
bash run_serve_bench.sh shape.yaml
```

默认测试 4 种请求形状、2 档 prefill size，共 54 个并发测试档位。H20 基线在 2026-10-15 17:21–20:49 跑完，约 **3 小时 28 分钟**；目标 GPU 的耗时以实测为准，首次下载和数据处理还需要额外时间。通过 SSH 运行时，建议放在 `tmux` 等可保持运行的终端会话里。

只想先测部分组合，可以执行：

```bash
bash run_serve_bench.sh shape.yaml --shapes short --prefill-sizes 4096
```

这仍会跑 `short` 的全部 9 个并发档位，也仍会准备两份数据、预热短请求和长请求；要做单并发试跑，请使用第 4 步的配置。

### 6. 查看并回传结果

每次运行使用独立的 `results/shape_<时间戳>/` 目录：

```text
results/shape_<时间戳>/
  serve_report.md                   吞吐、延迟、满足 SLO 的最大并发
  summary.jsonl                     每个测试档位一行的机器可读结果
  config.yaml  sampling_config.json 本次配置
  env.txt  serve.log                GPU、软件版本和完整运行日志
  ps4096/  psdefault/               各形状、各并发的逐请求 CSV
  figs/serve.png                    吞吐和延迟曲线
```

### 7. 怎样比较不同 GPU 的结果

已有参考基线见 [H20 请求形状 × 并发报告](reports/h20_prefill_request_shapes_2026-10-15.pdf)，其运行标识为 `shape_20261015_172145`。固定 prefill size 4096 的几个参照点如下：

| 请求形状 | 并发 | H20 计算 tok/s | H20 服务 tok/s | H20 p90 延迟 |
|---|---|---|---|---|
| `short`（约 4k） | 1 | 19,549 | 19,549 | 0.22 s |
| `short`（约 4k） | 128 | 23,402 | 23,402 | 24.4 s |
| `long`（约 64k） | 1 | 7,387 | 7,387 | 9.2 s |
| `long`（约 64k） | 32 | 8,370 | 8,370 | 291 s |
| `long_seg_cache`（每段） | 16 条轨迹 | 7,775 | 20,209 | 48 s |

- 对齐**请求形状、并发、prefill size、模型层数和软件版本**后比较。吞吐倍率 = 目标 GPU tok/s ÷ 参考 GPU tok/s；延迟越低越好。
- 优先比较固定 `4096` 那组。`default` 由 SGLang 按显存等条件选择，H20 上是 chunk 8192、`max_prefill_tokens` 16384，不同 GPU 不保证相同。
- 缓存测试同时看**服务 tok/s、缓存命中率和 KV 容量**。H20 基线显存为 141 GB，KV 池容量为 1,067,688 token；目标 GPU 的显存以 `nvidia-smi` 输出为准，KV 容量以引擎实际分配为准。缓存逐出的时机可能不同，服务吞吐差异包含显存容量的影响。
- H20 基线使用 DeepGEMM。目标 GPU 的实际 FP8 / attention 后端应记录在结果说明中；如果因硬件兼容性调整了后端或软件版本，比较时一并注明。
- 固定模型 `Qwen/Qwen3-8B-FP8`、`latent_layer: 23`、上下文长度 81920、采样规则、正式测试时长和并发档位。若为了适配硬件修改了这些设置，回传时注明，结果应单独比较。
- 低并发长请求的样本较少，p90 只作粗略参考；本测试不包含 HTTP 或输入分词时间，也不测多 token decode 性能。

## 常见问题

| 现象 | 处理方式 |
|---|---|
| 数据处理占用太多 CPU / 主机内存 | 将 `processing.workers: 0`（所有 CPU）改为较小的进程数，例如 8。 |
| 启动时长时间没有新输出 | 首次 kernel 编译和模型加载可能较慢；查看当前结果目录的 `serve.log`，终端只显示筛选后的进度行。 |
| 运行退出，但终端没有报错详情 | 查看本次结果目录的 `serve.log`；完整异常写在这里。 |
| CUDA / FP8 kernel 不支持，或显存不足 | 先确认该卡对当前 SGLang 和 FP8 模型的支持；提供 GPU 型号、显存及 `serve.log`。显存不足还需区分激活空间与 KV 池容量，改动参数后应注明。 |

## 请求形状测试的原理

下面是快速开始所运行测试的详细口径。

请求形状测试测量真实调度器下的吞吐与延迟，包括并发、KV cache 复用和 chunked prefill 的影响，由 [`run_serve_bench.sh`](run_serve_bench.sh) + [`scripts/serve_bench.py`](scripts/serve_bench.py) 负责，配置在 [`shape.yaml`](shape.yaml)：

```bash
bash run_serve_bench.sh                                   # 默认 shape.yaml
bash run_serve_bench.sh shape.yaml --shapes short --prefill-sizes 4096   # 只跑一部分
```

- **数据**：WildChat 中 3.5k–4.5k token 的对话 1 万条（短），SWE 轨迹中 56k–72k token 的 1 万条（长）。`prepare_data.py` 将原始记录转换为完整消息列表，在最后一次 assistant 回复之前截断，用模型的 chat template 计算 token 长度，再按固定 seed 和长度范围抽样。输出在 `data/processed_shape/`，长度索引保存在其中的 `.index/`，重复运行会复用。两批数据只作样本池，每个并发档位达到测量条件后停止发新请求，不会把每批 1 万条全跑完。
- **服务端**：进程内 SGLang Engine（真实调度器、continuous batching、chunked prefill、radix cache），直接传 token ids，`max_new_tokens=1`，所以 prefill 完就结束，没有 decode。`serve.latent_layer: 23` 表示只跑 decoder block 0..23（36 层的 2/3），模型按 24 层构建（`num_hidden_layers` 覆盖），权重和 KV 池也只按 24 层分配；final norm、lm_head 用 [`latent_exit.py`](scripts/latent_exit.py) 去掉，采样器只拿到一个全零 logits。
- **四种请求形状**：
  - `short`：独立的 ~4k 请求（radix cache 关）；
  - `long`：独立的 ~64k 请求，一次性 prefill（radix cache 关）；
  - `long_seg_cache`：一条长轨迹在最接近 1/4、2/4、3/4 长度的 assistant 轮次边界切开，变成 4 个累积前缀请求（约 16k / 32k / 48k / 64k），按顺序发送。radix cache 开，所以后一段会复用前一段的 KV（前提是还没被逐出）；
  - `long_seg_nocache`：同样的 4 段请求，但 radix cache 关，相当于前面轮次的 KV 已经被逐出，每一段都要把整个前缀重算一遍。
- **prefill size**：`serve.prefill_sizes: [4096, default]`。整数表示同时设 `chunked_prefill_size` 和 `max_prefill_tokens`，即一次前向最多处理多少 token；`default` 表示不设，由 SGLang 按显存自动选（H20：chunk 8192，max_prefill_tokens 16384）。
- **并发**：闭环，始终有 C 个请求（分段形状里是 C 条轨迹）在途。每档至少跑 60 秒，并且至少完成 4×C 个请求后停止发新请求；若先达到 900 秒上限，也会停止发新请求，然后等待在途请求完成。统计窗口从开始发请求到停止发请求为止，跨越窗口边界的请求按重叠比例计入。
- **指标**：计算 tok/s（输入 token 减去缓存命中，即 GPU 实际 prefill 的量）、服务 tok/s（全部输入 token）、每个请求的延迟 p50/p90/p99（客户端从提交到完成，含排队）。报告 `serve_report.md` 会列出每种形状、每个 prefill size 在各个延迟要求下（`serve.slo_seconds`，按 p90）能支持的最大并发。
- **首次启动**：模型加载、kernel 编译和预热会增加准备时间，具体取决于 GPU 和实际使用的后端。H20 基线使用 DeepGEMM，并缓存了编译结果；目标 GPU 首次运行的准备时间应单独观察。
