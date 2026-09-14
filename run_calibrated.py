import paramiko
import sys

host = '100.121.224.32'
user = 'niconicoluarte'
password = 'dw4yb26f'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=10)
sftp = client.open_sftp()

print("Uploading run_brms_calibrated.R...")
sftp.put("src/simulations/run_brms_calibrated.R", "cortico-cerebellar-model/run_brms_calibrated.R")

print("Running calibrated model...")
cmd = "cd cortico-cerebellar-model && Rscript run_brms_calibrated.R"
stdin, stdout, stderr = client.exec_command(cmd)

for line in stdout: print(line, end="")
for line in stderr: print(line, end="")

sftp.close()
client.close()
print("Done.")
