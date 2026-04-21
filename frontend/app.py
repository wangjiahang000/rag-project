import gradio as gr
from .api_client import APIClient

client = APIClient()

def chat_fn(message, history):
    formatted_history = []
    for h in history:
        formatted_history.append([h[0], h[1]])
    answer = client.chat(message, formatted_history)
    return answer

def upload_fn(file):
    if file is None:
        return "请选择文件"
    result = client.upload_file(file.name)
    if result.get("status") == "success":
        return f"✅ 上传成功，{result.get('chunks', 0)} 个文本块"
    return f"❌ 上传失败：{result.get('message', '未知错误')}"

def search_fn(query, max_results):
    if not query:
        return "请输入关键词", gr.update(choices=[]), []
    papers = client.search_arxiv(query, max_results)
    choices = [f"{p['title'][:60]}... ({p['published']})" for p in papers]
    return f"✅ 找到 {len(papers)} 篇论文", gr.update(choices=choices), papers

def add_fn(selected, papers):
    if not selected or not papers:
        return "请先搜索并选择论文"
    for p in papers:
        display = f"{p['title'][:60]}... ({p['published']})"
        if display == selected:
            result = client.add_paper(p['id'], p['title'], "manual")
            if result.get("status") == "success":
                return f"✅ 已添加，{result.get('chunks', 0)} 个文本块"
            return f"❌ 添加失败：{result.get('message')}"
    return "❌ 未找到选中的论文"

with gr.Blocks(title="RAG 助手") as demo:
    gr.Markdown("# 📚 个人智能问答助手")
    
    with gr.Tab("💬 对话"):
        gr.ChatInterface(fn=chat_fn)
        
        with gr.Row():
            file_input = gr.File(label="上传PDF/TXT", file_types=[".pdf", ".txt"])
            upload_btn = gr.Button("上传到知识库")
            upload_status = gr.Textbox(label="状态", interactive=False)
            upload_btn.click(fn=upload_fn, inputs=file_input, outputs=upload_status)
    
    with gr.Tab("🔍 文献检索"):
        with gr.Row():
            search_input = gr.Textbox(label="搜索关键词", placeholder="例如: large language model")
            max_results = gr.Slider(5, 20, 10, step=5, label="结果数量")
        search_btn = gr.Button("搜索")
        search_status = gr.Textbox(label="状态", interactive=False)
        dropdown = gr.Dropdown(label="选择论文", choices=[], interactive=True)
        papers_state = gr.State([])
        add_btn = gr.Button("添加到知识库")
        add_status = gr.Textbox(label="添加状态", interactive=False)
        
        search_btn.click(fn=search_fn, inputs=[search_input, max_results],
                        outputs=[search_status, dropdown, papers_state])
        add_btn.click(fn=add_fn, inputs=[dropdown, papers_state], outputs=add_status)

if __name__ == "__main__":
    demo.launch(server_port=7860)