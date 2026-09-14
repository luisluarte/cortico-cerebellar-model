import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
# Using https://cloud.r-project.org
stdin, stdout, stderr = client.exec_command('Rscript -e "install.packages(\'brms\', repos=\'https://cloud.r-project.org\')"')
for line in stdout: print(line, end="")
for line in stderr: print(line, end="")
