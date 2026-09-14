import paramiko
import os
import sys

host = '100.121.224.32'
user = 'niconicoluarte'
password = 'dw4yb26f'

print(f"Connecting to {host}...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=10)
sftp = client.open_sftp()

local_script = "src/simulations/run_brms.R"
remote_script = "cortico-cerebellar-model/run_brms.R"

print(f"Uploading {local_script} to {remote_script}...")
sftp.put(local_script, remote_script)

print("Executing Bayesian Logistic Model on remote...")
# CD into the root and run it
cmd = f"cd cortico-cerebellar-model && Rscript run_brms.R"
stdin, stdout, stderr = client.exec_command(cmd)

# Print output in real-time
for line in stdout:
    print(line, end="")
for line in stderr:
    print(line, end="", file=sys.stderr)

sftp.close()
client.close()
print("\nDone executing remote Bayesian Model.")
