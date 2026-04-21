#!/usr/bin/env python
"""
自动下载所需的嵌入模型到本地 models 目录。
如果网络不可用，请手动下载：
https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from huggingface_hub import snapshot_download

def main():
    model_path = settings.embedding_model
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print(f"目标路径: {model_path}")
    os.makedirs(model_path, exist_ok=True)

    try:
        print(f"正在从 HuggingFace 下载模型 {model_name} ...")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"]  # 忽略非必要文件，加快下载
        )
        print("✅ 模型下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n请手动从以下网址下载所有文件并放入上述目录：")
        print(f"https://huggingface.co/{model_name}/tree/main")
        print("\n所需文件：config.json, pytorch_model.bin (或 model.safetensors), tokenizer.json, modules.json 等")

if __name__ == "__main__":
    main()