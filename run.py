import logging
import subprocess
import sys
import time
import requests
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = os.path.dirname(os.path.abspath(__file__))

_NPM_PATHS = [
    r"C:\Program Files\nodejs\npm.cmd",
    r"C:\Program Files (x86)\nodejs\npm.cmd",
    os.path.expandvars(r"%AppData%\npm\npm.cmd"),
    os.path.expandvars(r"%ProgramFiles%\nodejs\npm.cmd"),
]


def _find_npm():
    """查找 npm 可执行文件路径"""
    if sys.platform != "win32":
        return "npm"
    for p in _NPM_PATHS:
        if os.path.isfile(p):
            return p
    # 最后尝试通过 where 查找
    try:
        result = subprocess.run(["where", "npm"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return None


def build_frontend():
    """构建前端静态资源"""
    npm = _find_npm()
    if not npm:
        print("未找到 Node.js/npm，请安装 Node.js (https://nodejs.org)")
        print("或手动构建: cd frontend && npm install && npm run build")
        return False

    frontend_dir = os.path.join(ROOT, "frontend")

    # 先检查 node_modules 是否存在
    if not os.path.isdir(os.path.join(frontend_dir, "node_modules")):
        print("正在安装前端依赖...")
        result = subprocess.run(
            [npm, "install"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            encoding="utf-8", errors="replace",
            shell=(sys.platform == "win32"),
        )
        if result.returncode != 0:
            print("npm install 失败:", result.stderr)
            return False

    dist_dir = os.path.join(frontend_dir, "dist")
    if os.path.isdir(dist_dir):
        import shutil
        shutil.rmtree(dist_dir)

    print("正在构建前端...")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=frontend_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        shell=(sys.platform == "win32"),
    )
    if result.returncode != 0:
        print("前端构建失败:", result.stderr)
        return False
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if "vite" in line or "built" in line or "assets" in line:
                print("  " + line)
    print("前端构建完成")
    return True


def run_backend():
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT,
    )


def wait_for_backend(timeout=90):
    print("等待后端启动...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get("http://localhost:8000/health", timeout=2)
            if resp.status_code == 200:
                print("\n后端已就绪！")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print("\n后端启动超时")
    return False


if __name__ == "__main__":
    print("=" * 50)
    print("正在构建前端...")
    print("=" * 50)
    if not build_frontend():
        print("警告: 前端构建失败，将仅启动后端 API")

    print("=" * 50)
    print("正在启动后端服务...")
    print("=" * 50)
    backend = run_backend()

    if not wait_for_backend():
        backend.terminate()
        sys.exit(1)

    print("\n" + "=" * 50)
    print("系统已启动！")
    print("访问地址：http://localhost:8000")
    print("API 文档：http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        backend.terminate()
        time.sleep(1)
        print("服务已终止")
