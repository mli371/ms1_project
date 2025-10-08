import os
import subprocess
import sys

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

env = os.environ.copy()
env["MS_ROOT"] = repo
env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")

cmd1 = [sys.executable, "-m", "ms1.scripts.ms1_runner", "--help"]
cmd2 = [sys.executable, os.path.join(repo, "ms1", "scripts", "ms1_runner.py"), "--help"]

for cmd in (cmd1, cmd2):
    subprocess.check_call(cmd, cwd=repo, env=env)

print("OK")
