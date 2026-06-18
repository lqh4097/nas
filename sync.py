import subprocess
import sys
from datetime import datetime

GIT = r"C:\Program Files\Git\bin\git.exe"
REPO = r"d:\NAS项目"


def run(cmd, check=True):
    result = subprocess.run([GIT] + cmd, cwd=REPO, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if check and result.returncode != 0:
        print("命令失败，已终止")
        sys.exit(1)
    return result


print("拉取远端更新...")
run(["pull", "origin", "main"])

status = run(["status", "--porcelain"]).stdout.strip()
if not status:
    print("无本地变更，已是最新")
    sys.exit(0)

run(["status", "--short"], check=False)

msg = input("提交说明（直接回车使用时间戳）: ").strip()
if not msg:
    msg = f"sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

run(["add", "-A"])
run(["commit", "-m", msg])
run(["push", "origin", "main"])
print("同步完成")
