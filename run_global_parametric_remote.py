import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/run_global_parametric.R", "cortico-cerebellar-model/run_global_parametric.R")
cmd = "cd cortico-cerebellar-model && Rscript run_global_parametric.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig58_Global_Posteriors.png", "Fig58_Global_Posteriors.png")
sftp.get("cortico-cerebellar-model/Fig59_Global_Conditional_Effects.png", "Fig59_Global_Conditional_Effects.png")
client.close()
