import subprocess
import sys
import time
import requests
import os

def run_backend():
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

def wait_for_backend(timeout=90):
    print("等待后端启动...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get("http://localhost:8000/health", timeout=2)
            if resp.status_code == 200:
                print("\n✅ 后端已就绪！")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print("\n❌ 后端启动超时")
    return False

def run_frontend():
    return subprocess.Popen(
        [sys.executable, "-m", "frontend.app"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

if __name__ == "__main__":
    print("=" * 50)
    print("正在启动后端服务...")
    print("=" * 50)
    backend = run_backend()
    
    if not wait_for_backend():
        backend.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("正在启动前端界面...")
    print("=" * 50)
    frontend = run_frontend()
    
    print("\n" + "=" * 50)
    print("🎉 系统已启动！")
    print("前端地址：http://localhost:7860")
    print("后端地址：http://localhost:8000")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n\n正在停止服务...")
        backend.terminate()
        frontend.terminate()
        time.sleep(1)
        print("✅ 服务已终止")