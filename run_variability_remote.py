import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/compute_variability.R", "cortico-cerebellar-model/compute_variability.R")
cmd = "cd cortico-cerebellar-model && Rscript compute_variability.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig67_Policy_Variability.png", "Fig67_Policy_Variability.png")
client.close()
