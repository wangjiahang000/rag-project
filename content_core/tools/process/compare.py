import os
from litellm import completion

def compare(docs_a: list[str], docs_b: list[str]) -> str:
    """对比两组文档，返回对比分析结果"""
    prompt = f"""
对比以下两组文档：

A组：
{chr(10).join(docs_a)}

B组：
{chr(10).join(docs_b)}

从核心观点、方法、结论三方面对比异同。
"""
    response = completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL")
    )
    return response.choices[0].message.content