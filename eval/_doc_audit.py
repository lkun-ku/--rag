"""文档一致性自动复检：标题层级 / 死链 / 代码路径引用 / 关键数字。

设计原则：**用可验证的事实校验，不靠目测**。
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["README.md", "CHANGELOG.md", "PROJECT_DESCRIPTION.md", "ROADMAP.md",
        "docs/RESUME_PROJECT.md", "docs/INTERVIEW_QA.md"]
problems = []


def bad(msg):
    problems.append(msg)


print("=" * 100)
print("1. ROADMAP 4.46 / 4.47 的标题层级")
print("=" * 100)
ls = (ROOT / "ROADMAP.md").read_text(encoding="utf-8").splitlines()
seg = []
in_seg = False
for i, l in enumerate(ls, 1):
    if re.match(r"^## 4\.4[67]", l):
        in_seg = True
    elif re.match(r"^## 4\.", l) and in_seg and "4.4" not in l[:8]:
        in_seg = False
    if in_seg and re.match(r"^#{2,4} ", l):
        seg.append((i, l))
for i, l in seg:
    print("  %5d| %s" % (i, l[:78]))
nums = [re.match(r"^#{2,4} (4\.4[67](?:\.\d+)?)", l).group(1)
        for _, l in seg if re.match(r"^#{2,4} 4\.4[67]", l)]
print()
print("  编号序列: %s" % nums)
# 检查 4.46 的子节连续、4.47 的子节连续
for parent in ("4.46", "4.47"):
    subs = sorted(int(n.split(".")[-1]) for n in nums
                  if n.startswith(parent + "."))
    if subs:
        expect = list(range(1, len(subs) + 1))
        ok = subs == expect
        print("  %s 子节 %s -> %s" % (parent, subs, "连续 ✅" if ok else "**跳号 ❌**"))
        if not ok:
            bad("ROADMAP %s 子节跳号: %s（应为 %s）" % (parent, subs, expect))

print()
print("=" * 100)
print("2. Markdown 相对链接是否指向真实文件")
print("=" * 100)
n_link = 0
for f in DOCS:
    t = (ROOT / f).read_text(encoding="utf-8")
    for m in re.finditer(r"\]\(([^)#:]+\.(?:md|json|py|txt))\)", t):
        tgt = m.group(1)
        if tgt.startswith("http"):
            continue
        n_link += 1
        p = (ROOT / f).parent / tgt
        if not p.exists() and not (ROOT / tgt).exists():
            bad("%s 死链: %s" % (f, tgt))
print("  检查 %d 个相对链接" % n_link)
print("  死链: %s" % ([x for x in problems if "死链" in x] or "无 ✅"))

print()
print("=" * 100)
print("3. 文档里提到的代码文件是否真实存在")
print("=" * 100)
missing = set()
for f in DOCS:
    t = (ROOT / f).read_text(encoding="utf-8")
    for m in re.finditer(r"`((?:RAG_PROJECT/)?(?:tools|utils|graph2|graph|documents|api|agent|llm_models)/[a-z_0-9]+\.py)`", t):
        rel = m.group(1)
        p1 = ROOT / rel
        p2 = ROOT / "RAG_PROJECT" / rel
        if not p1.exists() and not p2.exists():
            missing.add(rel)
print("  引用的代码文件: %d 个 distinct" % len(set(
    m.group(1) for f in DOCS
    for m in re.finditer(r"`((?:RAG_PROJECT/)?(?:tools|utils|graph2|graph|documents|api|agent|llm_models)/[a-z_0-9]+\.py)`",
                         (ROOT / f).read_text(encoding="utf-8")))))
print("  不存在的: %s" % (sorted(missing) or "无 ✅"))
for x in missing:
    bad("文档引用了不存在的代码文件: %s" % x)

print()
print("=" * 100)
print("4. 关键数字一致性")
print("=" * 100)
checks = {
    "20/20": ["PROJECT_DESCRIPTION.md", "CHANGELOG.md", "README.md", "docs/RESUME_PROJECT.md"],
    "7,416": ["PROJECT_DESCRIPTION.md", "CHANGELOG.md", "README.md", "docs/RESUME_PROJECT.md"],
    "11,999": ["PROJECT_DESCRIPTION.md", "CHANGELOG.md", "README.md", "docs/RESUME_PROJECT.md"],
}
for kw, files in checks.items():
    miss = [f for f in files if kw not in (ROOT / f).read_text(encoding="utf-8")]
    print("  %-8s 缺失于: %s" % (kw, miss or "无 ✅"))
    for f in miss:
        bad("%s 缺关键数字 %s" % (f, kw))

# 过时数字（只允许出现在 ROADMAP 历史小节 / CHANGELOG 历史条目）
stale = ["6,090", "5,176", "8,867", "10,113"]
for f in DOCS:
    if f == "ROADMAP.md":
        continue
    t = (ROOT / f).read_text(encoding="utf-8")
    hits = [s for s in stale if s in t]
    if hits:
        print("  %-26s 残留过时数字 %s" % (f, hits))
        if f == "CHANGELOG.md" and hits == ["5,176", "8,867"]:
            print("    （CHANGELOG 的 v1.0 历史条目，允许）")
        else:
            bad("%s 残留过时数字 %s" % (f, hits))

print()
print("=" * 100)
print("5. 关键表述是否仍是旧的")
print("=" * 100)
# 这些是之前修过的矛盾，不应再出现
forbidden = [
    ("graph2", "无（请求级状态）", "3.3 表曾写 graph2 无记忆"),
    ("reranker", "已否决", "7.1 曾写 reranker 已否决"),
    ("3 轮", "只注入最近", "INTERVIEW 曾写只注入 3 轮"),
]
for f in DOCS:
    t = (ROOT / f).read_text(encoding="utf-8")
    for a, b, desc in forbidden:
        if a in t and b in t and ("%s" % b) in t:
            # 只报「同一段距离很近」的情况，避免误报
            idx = t.find(b)
            if idx >= 0 and a in t[max(0, idx - 200):idx + 200]:
                print("  %-26s **可能仍有旧表述**: %s" % (f, desc))
                bad("%s 可能仍有旧表述: %s" % (f, desc))
print("  （无输出 = 未发现）")

print()
print("=" * 100)
if problems:
    print("发现 %d 个问题：" % len(problems))
    for p in problems:
        print("  - %s" % p)
    sys.exit(1)
print("全部通过：无跳号 / 无死链 / 无失效代码引用 / 数字一致")
