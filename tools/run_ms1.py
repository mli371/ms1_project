import os
import subprocess
import sys

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
env["MS_ROOT"] = repo
subprocess.check_call([sys.executable, "-m", "ms1.scripts.ms1_runner", *sys.argv[1:]], env=env, cwd=repo)
