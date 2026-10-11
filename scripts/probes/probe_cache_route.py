"""可行性验证 3：缓存命中潜力 与 路由规则短路覆盖率（零 LLM 成本）。

A. 缓存：对自动题集做两两语义相似度统计，估计「语义缓存(>0.95)」的上限。
   注意：题集是为评测而生成的不同问题，命中率必然偏低；这个数字给出的是
   **下限参考**，真实收益取决于线上流量的问题重复度。

B. 路由短路：统计问题中出现「源目录名 / 主题词」的比例，即规则可确定
   「走知识库」的覆盖率（省 1 次路由 LLM 往返）。

用法：python _probe_cache_route.py
"""

import hashlib
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_cases():
    p = ROOT / "eval" / "golden_auto.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_cache(cases):
    from rag_kb.llm_models.embeddings_model import bge_embedding

    qs = [c["q"] for c in cases]
    print("[缓存] 题数:", len(qs))

    # 精确缓存：归一化后完全相同的比例
    def norm(s):
        return re.sub(r"[\s，。？！,.?!:：；;\"'`]+", "", s).lower()

    hashes = [hashlib.sha1(norm(q).encode()).hexdigest() for q in qs]
    dup = len(hashes) - len(set(hashes))
    print("[缓存] 精确重复（归一化后相同）: %d 条 / %.1f%%" % (dup, dup / len(qs) * 100))

    # 语义缓存：采样 100 题算两两相似度
    sample = qs[: min(100, len(qs))]
    vecs = np.array(bge_embedding.embed_documents(sample), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    sim = vecs @ vecs.T
    iu = np.triu_indices_from(sim, k=1)
    pairs = sim[iu]
    print("[缓存] 两两相似度（%d 对）: 中位 %.3f，P90 %.3f，P99 %.3f" % (
        len(pairs), np.median(pairs), np.percentile(pairs, 90), np.percentile(pairs, 99)))
    for th in (0.90, 0.95, 0.98):
        n = int((pairs >= th).sum())
        print("[缓存] 相似度 >= %.2f 的对子: %d（%.2f%%）" % (th, n, n / len(pairs) * 100))


def probe_route(cases):
    from rag_kb.utils.env_utils import DOC_SOURCES

    names = [s["name"] for s in DOC_SOURCES if s.get("enabled")]
    # 目录名拆成主题词（中文不分词，退化为子串匹配）
    print("\n[路由] 当前源名:", names)

    hit = 0
    for c in cases:
        q = c["q"]
        if any(n in q for n in names) or any(n.replace("-", "") in q for n in names):
            hit += 1
    print("[路由] 问题中直接出现源名的比例: %d/%d = %.1f%%" % (hit, len(cases), hit / len(cases) * 100))

    # 更宽松：出现「资料中」这类明确指向知识库的措辞
    soft = sum(1 for c in cases if re.search(r"资料|文档|材料|手册|笔记", c["q"]))
    print("[路由] 含「资料/文档/材料」等指代的比例: %d/%d = %.1f%%" % (soft, len(cases), soft / len(cases) * 100))
    print("[路由] 两者并集: %.1f%%" % ((hit + soft - sum(1 for c in cases if (any(n in c["q"] for n in names)) and re.search(r"资料|文档|材料|手册|笔记", c["q"]))) / len(cases) * 100))


if __name__ == "__main__":
    cases = load_cases()
    probe_cache(cases)
    probe_route(cases)
