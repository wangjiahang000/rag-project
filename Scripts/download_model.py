#!/usr/bin/env python
"""
自动下载 BAAI/bge-m3 嵌入模型到本地 models 目录。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from huggingface_hub import snapshot_download

def main():
    model_path = settings.embedding_model
    model_name = "BAAI/bge-m3"

    print(f"目标路径: {model_path}")
    os.makedirs(model_path, exist_ok=True)

    try:
        print(f"正在从 HuggingFace 下载模型 {model_name} ...")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"]
        )
        print("✅ 模型下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"\n请手动从以下网址下载并放入 {model_path}：")
        print(f"https://huggingface.co/{model_name}")

if __name__ == "__main__":
    main()
