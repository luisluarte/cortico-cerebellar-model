import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')

client.exec_command('mkdir -p cortico-cerebellar-model/src/simulations')

sftp = client.open_sftp()
print("Uploading script to remote server...")
sftp.put("src/simulations/run_brms_adaptation_remote.R", "cortico-cerebellar-model/src/simulations/run_brms_adaptation_remote.R")
print("Uploaded. Starting execution...")

cmd = "cd cortico-cerebellar-model && Rscript src/simulations/run_brms_adaptation_remote.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)

for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()

client.close()
