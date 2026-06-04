#!/usr/bin/env python3
"""重索引脚本 — 用结构化分块重建向量库

用法：
    python scripts/reindex.py                             # 结构分块 + 索引已有 TXT
    python scripts/reindex.py --extract-pdfs              # 先提取 PDF → TXT，再索引
    python scripts/reindex.py --strategy recursive         # 递归字符分块（对比基线）
    python scripts/reindex.py --strategy structure         # 结构化分块（默认）
    python scripts/reindex.py --dry-run                   # 仅预览分块效果，不写入
    python scripts/reindex.py --clear                      # 清空旧数据后重建

示例：
    # 从 data/ 提取所有 PDF 并重建索引
    python scripts/reindex.py --extract-pdfs --clear

    # 仅索引已有 TXT（跳过 PDF 提取）
    python scripts/reindex.py --clear
"""

import io
import os
import re
import sys
import glob
import argparse
import logging

# Windows 下的 UTF-8 输出支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# HuggingFace 国内镜像（chroma SentenceTransformer 依赖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("reindex")

# ── 默认路径 ──
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
PAPERS_TXT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "last", "storage", "papers", "txt"
)
PAPERS_PDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data"
)


def dry_run_preview(txt_dir: str, strategy: str, max_files: int = 3):
    """预览分块效果，不写入向量库"""
    from content_core.data.chunker import chunk_text

    txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    if not txt_files:
        logger.error("未找到 TXT 文件: %s", txt_dir)
        return

    logger.info("TXT 目录: %s（共 %d 个文件）", txt_dir, len(txt_files))
    logger.info("分块策略: %s", strategy)
    logger.info("=" * 60)

    for fp in txt_files[:max_files]:
        filename = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        chunks = chunk_text(text, source=filename, strategy=strategy)

        # 提取章节分布
        chapters = {}
        for c in chunks:
            ch = c["metadata"]["chapter"] or "(meta)"
            chapters[ch] = chapters.get(ch, 0) + 1

        print(f"\n--- {filename} ---")
        print(f"   全文长度: {len(text)} 字符")
        print(f"   分块数:   {len(chunks)}")
        print(f"   章节分布: {chapters}")
        if chunks:
            print(f"   首块预览: {chunks[0]['text'][:120]}...")

    print("\n" + "=" * 60)
    print("预览结束。去掉 --dry-run 执行实际索引。")


# ── PDF 文本清洗 ──────────────────────────────

# 已知的噪声行（统一小写）
_NOISE_LINES = {
    "6202", "yam", "]lc.sc[", "1v77010.5062:vixra",
}

# 运行页眉正则（"论文标题 页码" — 纯字母开头，以数字结尾，无句号）
_RUNNING_HEADER = re.compile(r'^[A-Z][a-zA-Z\s]{10,70}\d+$')


def _is_noise_line(line: str) -> bool:
    """判断一行是否为 PDF 提取噪声"""
    s = line.strip()
    if not s:
        return False

    # 1. 精确匹配已知噪声
    if s.lower() in _NOISE_LINES:
        return True

    # 2. 反转文本特征：只含特殊字符 + 字母数字混合无空格
    if re.match(r'^[^a-zA-Z0-9\u4e00-\u9fff]{2,}$', s):
        return True  # 纯特殊字符行

    # 3. 反转的 arXiv ID（数字+字母+可选点号+冒号+逆序）
    if re.match(r'^\d+[a-z]+[\d.]*:\d+[a-z]+$', s.lower()):
        return True

    # 4. 运行页眉（论文标题 + 页码，每页顶部/底部重复）
    if _RUNNING_HEADER.match(s):
        return True

    # 5. 单独的年份反转（如 "6202" 已被精确匹配覆盖，此处兜底）
    if re.match(r'^\d{4}$', s) and 1900 <= int(s) <= 2099:
        return True

    # 6. 作者运行页眉："N 姓氏 et al."
    if re.match(r'^\d+ [A-Z][a-z]+ et al\.?$', s):
        return True

    return False


def _cleanup_page_text(text: str) -> str:
    """清理单页 PDF 提取文本中的噪声"""
    lines = text.split("\n")
    cleaned = [line for line in lines if not _is_noise_line(line)]
    return "\n".join(cleaned)


def extract_pdfs(pdf_dir: str, txt_dir: str, overwrite: bool = False) -> int:
    """将 PDF 批量提取为 TXT（跳过已存在的）

    Returns:
        新提取的文件数
    """
    import glob
    from pdfplumber import open as open_pdf

    os.makedirs(txt_dir, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        logger.error("未找到 PDF 文件: %s", pdf_dir)
        return 0

    new_count = 0
    for fp in pdf_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        # 取 PDF 文件名中 arxiv_id 部分（文件名格式: "2301.12345_title_hash.pdf"）
        arxiv_id = base.split("_")[0] if "_" in base else base
        txt_path = os.path.join(txt_dir, f"{arxiv_id}.txt")
        if os.path.exists(txt_path) and not overwrite:
            continue

        try:
            with open_pdf(fp) as pdf:
                pages = []
                for page in pdf.pages:
                    body = _cleanup_page_text(page.extract_text() or "")
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if not table or len(table) < 1:
                                continue
                            rows = []
                            for row in table:
                                cells = [c.replace("\n", " ") if c else "" for c in row]
                                rows.append("| " + " | ".join(cells) + " |")
                            header = rows[0]
                            sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
                            md = header + "\n" + sep
                            if len(rows) > 1:
                                md += "\n" + "\n".join(rows[1:])
                            body += "\n\n【表格】\n" + md
                    pages.append(body)
                text = "\n\n".join(pages)
        except Exception as e:
            logger.warning("PDF 提取失败 %s: %s", os.path.basename(fp), e)
            continue

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        new_count += 1
        if new_count % 200 == 0:
            logger.info("已提取 %d/%d PDF", new_count, len(pdf_files))

    logger.info("PDF 提取完成: 新增 %d 个 TXT（共 %d 个 PDF）", new_count, len(pdf_files))
    return new_count


def main():
    parser = argparse.ArgumentParser(description="用结构化分块重建向量库")
    parser.add_argument(
        "--strategy", default="structure",
        choices=["structure", "recursive"],
        help="分块策略（默认 structure）",
    )
    parser.add_argument(
        "--txt-dir", default=PAPERS_TXT_DIR,
        help=f"TXT 论文目录（默认 {PAPERS_TXT_DIR}）",
    )
    parser.add_argument(
        "--pdf-dir", default=PAPERS_PDF_DIR,
        help=f"PDF 论文目录（默认 {PAPERS_PDF_DIR}，仅 --extract-pdfs 时使用）",
    )
    parser.add_argument(
        "--extract-pdfs", action="store_true",
        help="先从 PDF 提取 TXT，再执行索引",
    )
    parser.add_argument(
        "--chroma-dir", default=CHROMA_DIR,
        help=f"向量库目录（默认 {CHROMA_DIR}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览分块效果，不写入向量库",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="清空现有向量库后重建",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="增量索引模式：仅处理新文件，不重建已有索引",
    )
    parser.add_argument(
        "--max-chunk-size", type=int, default=512,
        help="单块最大字符数（默认 512）",
    )
    parser.add_argument(
        "--only", type=str, default="",
        help="仅处理指定文件名（支持通配符如 2605.01*.txt）",
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="单次最多处理文件数（默认 0 = 全部处理）",
    )
    parser.add_argument(
        "--threads", type=int, default=0,
        help="限制 CPU 线程数（默认 0 = 不限制，设为 2 或 4 可降低 CPU 占比）",
    )
    args = parser.parse_args()

    if args.threads > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
            os.environ[var] = str(args.threads)
        logger.info("CPU 线程数限制为 %d", args.threads)

    if args.dry_run:
        dry_run_preview(args.txt_dir, args.strategy)
        return

    # 可选：从 PDF 提取 TXT
    if args.extract_pdfs:
        logger.info("正在从 PDF 提取文本...")
        n = extract_pdfs(args.pdf_dir, args.txt_dir)
        if n == 0 and not glob.glob(os.path.join(args.txt_dir, "*.txt")):
            logger.error("未提取到任何 TXT，终止")
            return
        logger.info("TXT 目录现有文件数: %d", len(glob.glob(os.path.join(args.txt_dir, "*.txt"))))

    # 导入 VectorStore
    from content_core.data.vector_store import VectorStore

    vs = VectorStore(
        persist_dir=args.chroma_dir,
        embedding_model="BAAI/bge-small-zh-v1.5",
    )

    if args.clear:
        logger.warning("清空现有向量库...")
        vs.clear()

    existing = vs.count()
    logger.info("当前向量库文档数: %d", existing)

    # 增量 / 全量索引
    if args.incremental and existing > 0:
        logger.info("增量索引模式: 仅处理新文件")
        result = vs.incremental_index(
            txt_dir=args.txt_dir,
            source_prefix="arxiv",
            chunk_strategy=args.strategy,
            max_chunk_size=args.max_chunk_size,
            max_files=args.max_files,
        )
    else:
        result = vs.load_and_index(
            txt_dir=args.txt_dir,
            source_prefix="arxiv",
            chunk_strategy=args.strategy,
            clear_existing=False,
            max_chunk_size=args.max_chunk_size,
            max_files=args.max_files,
        )

    # 报告
    after = vs.count()
    print("\n" + "=" * 60)
    print("  索引完成报告")
    print("=" * 60)
    if args.incremental:
        print(f"  模式:          增量索引")
        print(f"  总文件数:      {result['total_files']}")
        print(f"  新增文件数:    {result['new_files']}")
        print(f"  新增分块数:    {result['total_chunks']}")
    else:
        print(f"  模式:          全量重建")
        print(f"  处理文件数:    {result['total_files']}")
        print(f"  新增分块数:    {result['total_chunks']}")
    print(f"  当前总文档数:  {after}")
    print(f"  分块策略:      {args.strategy}")
    print(f"  最大块大小:    {args.max_chunk_size}")
    if args.incremental and result['new_files'] == 0:
        print("  [提示] 无新文件需要索引")
    print("=" * 60)


if __name__ == "__main__":
    main()
