import subprocess
import time
import sys
import requests
import os

def run_backend():
    """启动后端服务"""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

def wait_for_backend(timeout=90):  # 从 60 秒改为 90 秒
    """等待后端启动完成，最多等待 timeout 秒"""
    print("等待后端启动...")
    print("（模型加载需要 15-30 秒，请耐心等待）")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # 尝试连接后端健康检查接口
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("\n✅ 后端已就绪！")
                return True
        except requests.exceptions.ConnectionError:
            # 后端还没启动，继续等待
            pass
        except Exception:
            pass
        
        # 打印进度点
        print(".", end="", flush=True)
        time.sleep(2)
    
    print("\n❌ 后端启动超时（超过 {} 秒）".format(timeout))
    return False

def run_frontend():
    """启动前端服务"""
    return subprocess.Popen(
        [sys.executable, "frontend/app.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

if __name__ == "__main__":
    print("=" * 50)
    print("正在启动后端服务...")
    print("=" * 50)
    backend = run_backend()
    
    # 等待后端完全启动（增加到 90 秒）
    if not wait_for_backend(timeout=90):  # 修改这里
        print("后端启动失败，正在退出...")
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