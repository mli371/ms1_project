import os
import subprocess
import sys

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

env = os.environ.copy()
env["MS_ROOT"] = repo
env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")

cmds = [
    [sys.executable, "-m", "scripts.ms1_runner", "--help"],
    [sys.executable, os.path.join(repo, "scripts", "ms1_runner.py"), "--help"],
]

for cmd in cmds:
    subprocess.check_call(cmd, cwd=repo, env=env)

print("OK")
