import gradio as gr
import requests

def chat_with_deepseek(message, history):
    """多轮对话函数"""
    # 转换 Gradio 历史格式
    formatted_history = []
    i = 0
    while i < len(history):
        if history[i]['role'] == 'user' and i + 1 < len(history) and history[i+1]['role'] == 'assistant':
            user_text = history[i]['content'][0]['text'] if history[i]['content'] else ''
            assistant_text = history[i+1]['content'][0]['text'] if history[i+1]['content'] else ''
            formatted_history.append([user_text, assistant_text])
            i += 2
        else:
            i += 1
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "question": message,
                "history": formatted_history
            }
        )
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"错误: {response.status_code}"
    except Exception as e:
        return f"连接失败: {str(e)}"


def upload_file(file):
    """上传文档到知识库"""
    if file is None:
        return "请选择文件", gr.update(value=None)
    
    try:
        with open(file.name, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return f"✅ 上传成功！已处理 {result.get('chunks', 0)} 个文本块", gr.update(value=None)
            else:
                return f"❌ 上传失败：{result.get('message', '未知错误')}", gr.update(value=None)
        else:
            return f"❌ 上传失败：HTTP {response.status_code}", gr.update(value=None)
    except Exception as e:
        return f"❌ 连接失败：{str(e)}", gr.update(value=None)


def search_papers(query, max_results):
    """搜索arXiv论文"""
    if not query:
        return "请输入搜索关键词", gr.update(choices=[]), gr.update(value=[])
    
    try:
        response = requests.post(
            "http://localhost:8000/search_literature",
            json={"query": query, "max_results": max_results}
        )
        if response.status_code == 200:
            result = response.json()
            papers = result.get("papers", [])
            if papers:
                # 创建选项列表（显示中文标题）
                choices = []
                for p in papers:
                    # 使用翻译后的中文标题
                    title_display = p.get('title', p.get('title_en', '未知标题'))
                    # 限制长度
                    if len(title_display) > 80:
                        title_display = title_display[:77] + "..."
                    
                    authors_str = p.get('authors_display', ', '.join(p.get('authors', [])[:2]))
                    choices.append(f"{title_display} ({p['published']}) - {authors_str}")
                
                return f"✅ 找到 {len(papers)} 篇论文", gr.update(choices=choices), papers
            else:
                return "❌ 未找到相关论文", gr.update(choices=[]), []
        else:
            return f"❌ 搜索失败: HTTP {response.status_code}", gr.update(choices=[]), []
    except Exception as e:
        return f"❌ 连接失败: {str(e)}", gr.update(choices=[]), []


def add_paper_to_kb(selected_paper, papers_list):
    """添加论文到知识库"""
    if not selected_paper or not papers_list:
        return "请先选择一篇论文"
    
    # 找到选中的论文
    for paper in papers_list:
        # 构建匹配字符串
        title_display = paper.get('title', paper.get('title_en', '未知标题'))
        if len(title_display) > 80:
            title_display = title_display[:77] + "..."
        authors_str = paper.get('authors_display', ', '.join(paper.get('authors', [])[:2]))
        display_text = f"{title_display} ({paper['published']}) - {authors_str}"
        
        if display_text == selected_paper:
            try:
                response = requests.post(
                    "http://localhost:8000/add_paper_to_kb",
                    json={"paper_id": paper['id'], "paper_title": paper.get('title', paper.get('title_en', '未知'))}
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        return f"✅ {result.get('message', '添加成功')}，共 {result.get('chunks', 0)} 个文本块"
                    else:
                        return f"❌ 添加失败：{result.get('message', '未知错误')}"
                else:
                    return f"❌ 添加失败：HTTP {response.status_code}"
            except Exception as e:
                return f"❌ 连接失败: {str(e)}"
    
    return "❌ 未找到选中的论文"
def add_paper_to_kb(selected_paper, papers_list):
    """添加论文到知识库"""
    if not selected_paper or not papers_list:
        return "请先选择一篇论文"
    
    # 找到选中的论文
    for paper in papers_list:
        authors_str = ', '.join(paper['authors'][:2])
        if len(paper['authors']) > 2:
            authors_str += ' et al.'
        display_text = f"{paper['title']} ({paper['published']}) - {authors_str}"
        
        if display_text == selected_paper:
            try:
                response = requests.post(
                    "http://localhost:8000/add_paper_to_kb",
                    json={"paper_id": paper['id'], "paper_title": paper['title']}
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        return f"✅ {result.get('message', '添加成功')}，共 {result.get('chunks', 0)} 个文本块"
                    else:
                        return f"❌ 添加失败：{result.get('message', '未知错误')}"
                else:
                    return f"❌ 添加失败：HTTP {response.status_code}"
            except Exception as e:
                return f"❌ 连接失败: {str(e)}"
    
    return "❌ 未找到选中的论文"


# 创建界面
with gr.Blocks(title="个人智能问答助手", theme="soft") as demo:
    gr.Markdown("# 📚 个人智能问答助手")
    gr.Markdown("上传文档或搜索arXiv论文，然后基于内容提问")
    
    with gr.Tabs():
        # Tab 1: 对话
        with gr.TabItem("💬 对话"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.ChatInterface(
                        fn=chat_with_deepseek,
                        title="对话区",
                        description="输入你的问题，我会记住之前的对话"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 文档上传")
                    file_input = gr.File(label="选择PDF或TXT文件", file_types=[".pdf", ".txt"])
                    upload_btn = gr.Button("上传到知识库")
                    upload_status = gr.Textbox(label="状态", interactive=False)
                    
                    upload_btn.click(
                        fn=upload_file,
                        inputs=file_input,
                        outputs=[upload_status, file_input]
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 💡 使用说明")
                    gr.Markdown("1. 上传PDF或TXT文档")
                    gr.Markdown("2. 等待处理完成")
                    gr.Markdown("3. 在对话框提问文档内容")
        
        # Tab 2: 文献检索
        with gr.TabItem("🔍 文献检索"):
            gr.Markdown("### 搜索arXiv学术论文")
            gr.Markdown("支持计算机科学、物理、数学等领域的论文检索")
            
            with gr.Row():
                search_input = gr.Textbox(
                    label="搜索关键词", 
                    placeholder="例如: large language model, attention mechanism, deep learning",
                    scale=3
                )
                max_results = gr.Slider(
                    label="结果数量", 
                    minimum=5, 
                    maximum=20, 
                    value=10, 
                    step=5, 
                    scale=1
                )
            
            search_btn = gr.Button("🔍 搜索论文", variant="primary")
            search_status = gr.Textbox(label="状态", interactive=False)
            
            paper_dropdown = gr.Dropdown(label="选择论文", choices=[], interactive=True)
            papers_state = gr.State([])
            
            add_btn = gr.Button("📥 添加到知识库", variant="secondary")
            add_status = gr.Textbox(label="添加状态", interactive=False)
            
            # 绑定事件
            search_btn.click(
                fn=search_papers,
                inputs=[search_input, max_results],
                outputs=[search_status, paper_dropdown, papers_state]
            )
            
            add_btn.click(
                fn=add_paper_to_kb,
                inputs=[paper_dropdown, papers_state],
                outputs=[add_status]
            )
            
            gr.Markdown("---")
            gr.Markdown("### 💡 使用说明")
            gr.Markdown("1. 输入关键词搜索arXiv论文")
            gr.Markdown("2. 从下拉列表选择感兴趣的论文")
            gr.Markdown("3. 点击「添加到知识库」下载并导入")
            gr.Markdown("4. 切换到「对话」Tab提问论文内容")
            gr.Markdown("5. 注意：arXiv API有频率限制，请勿频繁搜索")

# 启动应用
demo.launch(server_port=7860)