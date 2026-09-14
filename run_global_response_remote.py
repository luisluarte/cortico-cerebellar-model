import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/plot_global_response_scale.R", "cortico-cerebellar-model/plot_global_response_scale.R")
cmd = "cd cortico-cerebellar-model && Rscript plot_global_response_scale.R"
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
sftp.get("cortico-cerebellar-model/Fig60_Global_Response_Dynamics.png", "Fig60_Global_Response_Dynamics.png")
client.close()
