import gradio as gr
import requests

def chat_with_deepseek(message,history):
    try:
        response=requests.post(
            "http://localhost:8000/chat",
            json={"question":message}
        )
        if response.status_code==200:
             return response.json()["answer"]
        else:
            return f"错误:{response.status_code}"
    except Exception as e:
        return f"连接失败:{str(e)}"
    
def upload_file(file):
    if file is None:
        return "请选择文件",gr.update(value=None)
    
    try:
        with open(file.name,'rb')as f:
            files = {'file':f}
            response =requests.post("http://localhost:8000/upload",files=files)

        if response.status_code==200:
            result=response.json()
            if result.get("status")=="success":
                return f"帅 上传成功!已处理{result.get('chunks',0)}个文本块",gr.update(value=None)
            else:
                return f"可恶 上传失败：{result.get('message','未知错误')}",gr.update(value=None)
        else:
            return f"可恶 上传失败：HTTP{response.status_code}",gr.update(value=None)
    except Exception as e:
        return f"可恶 连接失败：{str(e)}",gr.update(value=None)


with gr.Blocks(title="个人智能问答助手") as demo:
    gr.Markdown("# 📚 个人智能问答助手")
    gr.Markdown("上传文档后，可以基于文档内容提问")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat_with_deepseek,
                title="对话区",
                description="输入你的问题"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📁 文档上传")
            file_input = gr.File(label="选择PDF或TXT文件", file_types=[".pdf", ".txt"])
            upload_btn = gr.Button("上传到知识库")
            upload_status = gr.Textbox(label="状态", interactive=False)
            
            upload_btn.click(
                fn=upload_file,
                inputs=file_input,
                outputs=[upload_status,file_input]
            )
            
            gr.Markdown("---")
            gr.Markdown("### 💡 使用说明")
            gr.Markdown("1. 上传PDF或TXT文档")
            gr.Markdown("2. 等待处理完成")
            gr.Markdown("3. 在对话框提问文档内容")

demo.launch(server_port=7860)