import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/plot_correction_iqr.R", "cortico-cerebellar-model/plot_correction_iqr.R")
cmd = "cd cortico-cerebellar-model && Rscript plot_correction_iqr.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig64_Cerebellar_Correction_IQR.png", "Fig64_Cerebellar_Correction_IQR.png")
client.close()
