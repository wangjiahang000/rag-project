import subprocess
import time
import sys

def run_backend():
    return subprocess.Popen(
        [sys.executable,"-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000","--reload"]
    )

def run_frontend():
    return subprocess.Popen(
        [sys.executable,"frontend/app.py"]
    )

if __name__=="__main__":
    print("正在启动后端服务...")
    backend=run_backend()
    time.sleep(3)

    print("正在启动前端界面...")
    frontend=run_frontend()

    print("="*50)
    print("系统已启动！")
    print("前端地址：http://localhost:7860")
    print("后端地址：http://localhost:8000")
    print("按Ctrl+C 停止服务")
    print("="*50)


    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        backend.terminate()
        frontend.terminate()
        time.sleep(0.5)
        print("服务已终止")