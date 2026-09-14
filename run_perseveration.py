import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/build_perseveration_data.R", "cortico-cerebellar-model/build_perseveration_data.R")
cmd = "cd cortico-cerebellar-model && Rscript build_perseveration_data.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/results/perseveration_df.rds", "results/perseveration_df.rds")
client.close()
