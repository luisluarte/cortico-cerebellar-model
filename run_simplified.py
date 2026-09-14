import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/run_brms_calibrated_simplified.R", "cortico-cerebellar-model/run_brms_calibrated_simplified.R")
cmd = "cd cortico-cerebellar-model && Rscript run_brms_calibrated_simplified.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
client.close()
