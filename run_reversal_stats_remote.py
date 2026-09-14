import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/calc_reversal_stats.R", "cortico-cerebellar-model/calc_reversal_stats.R")
cmd = "cd cortico-cerebellar-model && Rscript calc_reversal_stats.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig57_Reversal_Contrast.png", "Fig57_Reversal_Contrast.png")
client.close()
