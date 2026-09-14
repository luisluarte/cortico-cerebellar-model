import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/r/sim_bio_probes.cpp", "cortico-cerebellar-model/src/r/sim_bio_probes.cpp")
sftp.put("src/simulations/build_reversal_data.R", "cortico-cerebellar-model/build_reversal_data.R")
cmd = "cd cortico-cerebellar-model && Rscript build_reversal_data.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/results/reversal_df.rds", "results/reversal_df.rds")
client.close()
