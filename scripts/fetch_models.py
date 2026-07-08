"""下载嵌入模型到项目内，使模型跟随项目走（models/ 不进 git，此脚本负责重建）。

为什么需要这个脚本：
    `models/` 已被 .gitignore 排除（bge-large-zh-v1.5 约 1.3 GB，提交进 git 会让
    仓库无法使用），因此换机/克隆后需要重新获取模型。本脚本把获取过程固化下来，
    避免依赖「记得手动下过什么」。

用法：
    # 默认：下载 bge-large-zh-v1.5 到 RAG_PROJECT/models/（等价于项目默认配置）
    .venv/Scripts/python.exe scripts/fetch_models.py

    # 指定模型与目标目录
    .venv/Scripts/python.exe scripts/fetch_models.py --model BAAI/bge-small-zh-v1.5

    # 本机有 HTTPS 代理做 TLS 拦截（代理 CA 不在 certifi 中）时
    .venv/Scripts/python.exe scripts/fetch_models.py --insecure

为什么默认用 ModelScope 源（实测数据，同一时刻同一网络）：
    | 源                | 状态            | 速度        | 1.3 GB 预计耗时 |
    |-------------------|-----------------|-------------|-----------------|
    | huggingface.co    | ConnectTimeout  | —           | 不可用          |
    | hf-mirror.com     | 可下载          | ~64 KB/s    | 约 5.6 小时     |
    | modelscope.cn     | 可下载          | ~14 MB/s    | 约 1.5 分钟     |

两个必须知道的坑（脚本已处理，换源时请注意）：
  1. **ModelScope 的文件列表与 HuggingFace 不同** —— 它给出 `configuration.json`，
     但**缺少 `1_Pooling/config.json`**。缺这个文件会导致加载时报
     `TypeError: Pooling.__init__() missing 1 required positional argument:
     'embedding_dimension'`。脚本会从 hf-mirror 自动补下该文件。
  2. **bge-large-zh-v1.5 只有 `pytorch_model.bin`，没有 `.safetensors`**。
     任何 `ignore_patterns=["*.bin"]` 的配置都会排除掉唯一的权重文件。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import urllib3
import requests

DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
HF_MIRROR = "https://hf-mirror.com"
MODELSCOPE = "https://www.modelscope.cn"
# 项目根（scripts/ 的上一级）
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DEST_ROOT = ROOT / "RAG_PROJECT" / "models"

# ModelScope 已知缺失、需要从 hf-mirror 补齐的文件（相对路径）
MISSING_ON_MODELSCOPE = ("1_Pooling/config.json",)


def make_session(insecure: bool) -> requests.Session:
    """创建会话。--insecure 时关闭证书校验（仅用于本机代理做 TLS 拦截的场景）。"""
    s = requests.Session()
    if insecure:
        urllib3.disable_warnings()
        s.verify = False
        print("[警告] 已关闭 TLS 证书校验（--insecure）——仅应在受信任的本机代理下使用")
    return s


def list_files_modelscope(session, model: str):
    url = f"{MODELSCOPE}/api/v1/models/{model}/repo/files?Revision=master&Root="
    r = session.get(url, timeout=30)
    r.raise_for_status()
    files = (r.json().get("Data") or {}).get("Files") or []
    out = []
    for f in files:
        p = f.get("Path") or f.get("Name")
        if not p or f.get("Type") == "tree":
            continue
        out.append((p, f.get("Size") or 0))
    return out


def download_file(session, url: str, target: pathlib.Path, expect: int = 0, quiet: bool = False):
    """流式下载，支持断点续传（已存在且大小达标则跳过）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    have = target.stat().st_size if target.exists() else 0
    if expect and have >= expect:
        if not quiet:
            print(f"  SKIP {target.name} (已存在 {have / 1e6:.2f} MB)")
        return True

    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        r = session.get(url, timeout=60, stream=True, headers=headers)
        if r.status_code not in (200, 206):
            print(f"  FAIL {target.name} HTTP {r.status_code}")
            return False
        mode = "ab" if (have and r.status_code == 206) else "wb"
        if mode == "wb":
            have = 0
        t0 = time.time()
        got = have
        with open(target, mode) as fh:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    fh.write(chunk)
                    got += len(chunk)
        dt = max(time.time() - t0, 0.01)
        speed = (got - have) / dt / 1024
        print(f"  OK   {target.name:<34} {got / 1e6:8.2f} MB  {speed:6.0f} KB/s")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {target.name}: {type(e).__name__}: {str(e)[:80]}")
        return False


def fetch(model: str, dest_root: pathlib.Path, insecure: bool, source: str) -> int:
    session = make_session(insecure)
    name = model.split("/")[-1]
    dest = dest_root / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"模型     : {model}")
    print(f"目标目录 : {dest}")
    print(f"源       : {source}")
    print("-" * 74)

    ok = fail = 0
    t0 = time.time()

    if source == "modelscope":
        try:
            files = list_files_modelscope(session, model)
        except Exception as e:  # noqa: BLE001
            print(f"获取文件列表失败: {type(e).__name__}: {str(e)[:120]}")
            return 2
        print(f"文件数   : {len(files)}")
        for rel, size in files:
            url = f"{MODELSCOPE}/api/v1/models/{model}/repo?Revision=master&FilePath={rel}"
            target = dest / rel
            # 与 hf 的目录命名保持一致的已知差异：ModelScope 的 configuration.json 不是
            # sentence-transformers 需要的文件，跳过以免混淆。
            if rel == "configuration.json" and not (dest / "config.json").exists():
                target = dest / "config.json"
            if download_file(session, url, target, size):
                ok += 1
            else:
                fail += 1
    else:
        base = f"{HF_MIRROR}/{model}/resolve/main"
        for rel in ("config.json", "config_sentence_transformers.json", "modules.json",
                    "sentence_bert_config.json", "special_tokens_map.json",
                    "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                    "pytorch_model.bin"):
            if download_file(session, f"{base}/{rel}", dest / rel):
                ok += 1
            else:
                fail += 1

    # 补齐 ModelScope 缺失的文件（会导致加载失败，必须补）
    for rel in MISSING_ON_MODELSCOPE:
        target = dest / rel
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"  [补齐] {rel} 在 {source} 源缺失，从 hf-mirror 获取")
        if download_file(session, f"{HF_MIRROR}/{model}/resolve/main/{rel}", target):
            ok += 1
        else:
            fail += 1

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(dest) for f in fs)
    print("-" * 74)
    print(f"完成: 成功 {ok} / 失败 {fail} | 耗时 {time.time() - t0:.0f}s | 体积 {total / 1e6:.1f} MB")
    print(f"目录: {dest}")
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser(description="下载嵌入模型到项目内")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 id（默认 {DEFAULT_MODEL}）")
    ap.add_argument("--dest", default=str(DEFAULT_DEST_ROOT), help="模型根目录")
    ap.add_argument("--source", default="modelscope", choices=["modelscope", "hf-mirror"],
                    help="下载源（实测 modelscope 快约 200 倍）")
    ap.add_argument("--insecure", action="store_true",
                    help="关闭 TLS 证书校验（本机代理做 TLS 拦截时需要）")
    args = ap.parse_args()
    return fetch(args.model, pathlib.Path(args.dest), args.insecure, args.source)


if __name__ == "__main__":
    sys.exit(main())
