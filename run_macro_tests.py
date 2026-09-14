import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/macro_tests_remote.R", "cortico-cerebellar-model/macro_tests_remote.R")
cmd = "cd cortico-cerebellar-model && Rscript macro_tests_remote.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/results/long_df_all.rds", "results/long_df_all.rds")
client.close()
