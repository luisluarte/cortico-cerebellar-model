import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
stdin, stdout, stderr = client.exec_command('cat cortico-cerebellar-model/src/simulations/build_reversal_data.R')
for line in stdout: print(line, end='')
client.close()
