import paramiko
import sys

host = '100.121.224.32'
user = 'niconicoluarte'
password = 'dw4yb26f'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=10)
sftp = client.open_sftp()

remote_file = "cortico-cerebellar-model/results/brms_switch_summary.txt"
local_file = "results/brms_switch_summary.txt"

print(f"Downloading {remote_file} to {local_file}...")
sftp.get(remote_file, local_file)

sftp.close()
client.close()
print("Download complete.")
