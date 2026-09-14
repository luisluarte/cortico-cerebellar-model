import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/compute_wsls.R", "cortico-cerebellar-model/compute_wsls.R")
cmd = "cd cortico-cerebellar-model && Rscript compute_wsls.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig68_WSLS_Similarity.png", "Fig68_WSLS_Similarity.png")
client.close()
