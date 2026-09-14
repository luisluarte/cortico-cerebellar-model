import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/run_reversal_timecourse.R", "cortico-cerebellar-model/run_reversal_timecourse.R")
cmd = "cd cortico-cerebellar-model && Rscript run_reversal_timecourse.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig66_Reversal_Timecourse_GAM.png", "Fig66_Reversal_Timecourse_GAM.png")
client.close()
