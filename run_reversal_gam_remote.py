import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/run_reversal_gam.R", "cortico-cerebellar-model/run_reversal_gam.R")
cmd = "cd cortico-cerebellar-model && Rscript run_reversal_gam.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/results/brms_reversal_gam_ce.rds", "results/brms_reversal_gam_ce.rds")
client.close()
