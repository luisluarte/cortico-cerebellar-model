import paramiko
import os

host = '100.121.224.32'
user = 'niconicoluarte'
password = 'dw4yb26f'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=10)
sftp = client.open_sftp()

files_to_sync = [
    "src/r/sim_bio_probes.cpp",
    "src/r/distillation_targets_v2.rds",
    "results/factorial_fixed_10k_A.rds"
]

for f in files_to_sync:
    local_path = f
    remote_path = f"cortico-cerebellar-model/{f}"
    print(f"Uploading {local_path} to {remote_path}...")
    try:
        sftp.put(local_path, remote_path)
    except Exception as e:
        print(f"Failed to upload {f}: {e}")
        # Make directory if missing
        dir_name = os.path.dirname(remote_path)
        client.exec_command(f"mkdir -p {dir_name}")
        sftp.put(local_path, remote_path)

print("Running brms model again...")
cmd = "cd cortico-cerebellar-model && Rscript run_brms.R"
stdin, stdout, stderr = client.exec_command(cmd)

for line in stdout: print(line, end="")
for line in stderr: print(line, end="")

sftp.close()
client.close()
