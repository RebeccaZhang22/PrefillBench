# 数据下载

需要下载两个 Hugging Face 数据集，放在本目录（`data/`）下。

```bash
pip install -U huggingface_hub
# export HF_ENDPOINT=https://hf-mirror.com      # 国内网络可以走镜像
hf download allenai/WildChat-1M --repo-type dataset \
    --revision 7d6490e462285cf85d91eabea0f9a954fbddcd1f --local-dir data/WildChat-1M
hf download nebius/SWE-rebench-openhands-trajectories --repo-type dataset \
    --revision 35455389ab51bf5e2306bfd436ef72d0f98bf882 --local-dir data/SWE-rebench-openhands-trajectories
```

下载完成后，目录结构应该是这样的（大约 5.2 GB）：

```
data/WildChat-1M/data/train-000{00..13}-of-00014.parquet       14 个分片，共 837,989 行
data/SWE-rebench-openhands-trajectories/trajectories.parquet  67,074 行
```

- 老版本的 `huggingface_hub` 没有 `hf` 命令，可以把 `hf download` 换成 `huggingface-cli download`，参数一样。
- 如果数据放在别的位置，请修改 `config.yaml` 里各个 source 的 `input` 路径。
- 两个数据集的许可证：WildChat-1M 为 ODC-BY，SWE-rebench-openhands-trajectories 为 CC-BY-4.0。

下载好以后，运行 `bash run_pipeline.sh`。第 1 步会把原始数据处理并下采样，在 `data/processed/` 下生成 `wildchat.jsonl` 和 `swe_rebench.jsonl`。
