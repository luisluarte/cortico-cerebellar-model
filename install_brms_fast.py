import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')

# Create Makevars
stdin, stdout, stderr = client.exec_command('mkdir -p ~/.R && echo "CXX14FLAGS += -O3 -fPIC -Wno-ignored-attributes -Wno-deprecated-declarations -fno-lto\nCXX11FLAGS += -O3 -fPIC -Wno-ignored-attributes -Wno-deprecated-declarations -fno-lto\nCXXFLAGS += -O3 -fPIC -Wno-ignored-attributes -Wno-deprecated-declarations -fno-lto\nMAKEFLAGS = -j32" > ~/.R/Makevars')
stdout.read()

print("Makevars created. Starting parallel install...")

# Install brms with 32 cores
r_cmd = """
options(Ncpus = 32)
install.packages(c('RcppEigen', 'StanHeaders', 'rstan', 'brms'), repos='https://cloud.r-project.org', Ncpus=32)
"""
stdin, stdout, stderr = client.exec_command(f"Rscript -e \"{r_cmd}\"")

for line in stdout: print(line, end="")
for line in stderr: print(line, end="", file=sys.stderr)
