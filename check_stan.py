import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
stdin, stdout, stderr = client.exec_command('ls -l /tmp/Rtmp*/*.csv')
files = stdout.read().decode().strip().split('\n')
for f in files:
    if f.strip():
        filepath = f.split()[-1]
        print(f"--- {filepath} ---")
        std2, out2, err2 = client.exec_command(f'wc -l {filepath}')
        print(out2.read().decode())
