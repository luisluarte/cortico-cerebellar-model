import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.put("src/simulations/run_brms_raw_v2.R", "cortico-cerebellar-model/run_brms_raw_v2.R")
cmd = "cd cortico-cerebellar-model && Rscript run_brms_raw_v2.R"

# Adding get_pty=True forces Rscript to run interactively and immediately flush its stdout buffer!
stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)

# We loop and print. Because we are in a pseudo-terminal, it will stream the MCMC chain progress directly to our log!
for line in iter(stdout.readline, ""):
    print(line, end="")
    
client.close()
