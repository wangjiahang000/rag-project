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
    
demo=gr.ChatInterface(
    fn=chat_with_deepseek,
    title="个人智能问答助手",
    description="输入问题，DeepSeek会为你解答"
    )

demo.launch(server_port=7860)