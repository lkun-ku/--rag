"""LLM 端点可达性检查 —— 供 `run_all_checks.py` 做**前置门禁**。

## 为什么必须走 SDK 而不是裸 socket（测量口径的教训）

[] 排查 A/B 全超时的时候，我用 `socket.create_connection(('api.deepseek.com', 443))`
直连测，得出「端点不可达」的结论。**但这个口径是错的**：

* 裸 socket = **直连**，不考虑任何代理；
* 程序实际走的是 **openai SDK**，它会读 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`
  等环境变量。

**两条路径可能得出相反结论**（直连不通但走代理通，或反之）。
所以本检查**刻意独立构造一个 SDK client**，走与生产**同一条路径**来做探测。

## 顺带解决「白跑」问题

A/B 那次是**跑了 8 分钟**才在日志里发现全是超时。本检查作为前置门禁，
能在**十几秒内**给出「端点能不能用」，并把诊断信息（base_url / model / 代理环境变量）
一并打印出来 —— 出问题时不用再去翻日志。

用法：
    .venv\\Scripts\\python.exe eval\\check_llm.py              # 默认 20s 超时
    .venv\\Scripts\\python.exe eval\\check_llm.py --timeout 8   # 快速档用
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 端点可达性检查")
    ap.add_argument("--timeout", type=float, default=20.0, help="单次调用超时（秒）")
    args = ap.parse_args()

    from utils.env_utils import LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY

    proxies = {k: os.environ[k] for k in PROXY_KEYS if k in os.environ}
    print("  base_url : %s" % LLM_BASE_URL)
    print("  model    : %s" % LLM_MODEL)
    print("  api_key  : %s" % ("已配置" if OPENAI_API_KEY else "**缺失**"))
    print("  代理环境变量: %s" % (proxies if proxies else "（无 —— SDK 将直连）"))
    print("  超时     : %.0fs" % args.timeout)

    t0 = time.time()
    try:
        # 独立构造 client（带可控超时），走**与生产同一条 SDK 路径**
        from openai import OpenAI

        # **必须 `max_retries=0`**：openai SDK 默认重试 2 次，会让「12s 超时」
        # 实际耗时 ~75s（12s x 3 + 连接开销）—— 实测踩到，本检查的全部意义就是「快」。
        client = OpenAI(
            base_url=LLM_BASE_URL, api_key=OPENAI_API_KEY or "sk-placeholder", timeout=args.timeout, max_retries=0
        )
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "回复一个字：好"}],
        )
        reply = (r.choices[0].message.content or "").strip()
        print()
        print("  结果: **可用** —— OK (%r)，耗时 %.1fs" % (reply[:20], time.time() - t0))
        return 0
    except Exception as e:  # noqa: BLE001
        print()
        print("  结果: **不可用** —— %s: %s（耗时 %.1fs）" % (type(e).__name__, str(e)[:150], time.time() - t0))
        print()
        print("  排查建议：")
        print("    1) 该端点是否需要**代理**？SDK 只读上面列出的环境变量；")
        print("       若你的代理客户端监听在某个本地端口，请设 HTTPS_PROXY=http://127.0.0.1:<port> 再跑。")
        print(
            "    2) 注意 `.env` 里的 LLM_BASE_URL 优先级高于进程环境变量（load_dotenv 为覆盖模式），改端点要改 `.env`。"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
