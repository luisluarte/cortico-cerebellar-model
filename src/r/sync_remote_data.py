import os
import sys
import subprocess

# Auto-install paramiko if missing
try:
    import paramiko
except ImportError:
    print("Installing paramiko...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
    import paramiko

def pull_files():
    host = '100.121.224.32'
    user = 'niconicoluarte'
    password = 'dw4yb26f'
    
    # Get local repo root (assuming script is in src/r)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    print(f"Target Local Repo Root: {local_repo_root}")
    print("Connecting to remote Arch server...")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=10)
    
    sftp = client.open_sftp()
    
    # Find all data/model files that are typically gitignored
    cmd = 'cd ~/cortico-cerebellar-model && find . -type f \\( -name "*.rds" -o -name "*.csv" -o -name "*.pt" -o -name "*.json" -o -name "*.log" \\)'
    stdin, stdout, stderr = client.exec_command(cmd)
    files = stdout.read().decode().splitlines()
    
    print(f"Found {len(files)} matching files on remote.")
    
    downloaded = 0
    skipped = 0
    
    for remote_file in files:
        # remote_file looks like "./data/stan_data.json"
        if remote_file.startswith('./'):
            clean_path = remote_file[2:]
        else:
            clean_path = remote_file
            
        local_path = os.path.join(local_repo_root, os.path.normpath(clean_path))
        remote_full_path = f"cortico-cerebellar-model/{clean_path}"
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Check if local file exists and sizes match to skip unnecessary downloads (resume support)
        try:
            remote_stat = sftp.stat(remote_full_path)
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                if local_size == remote_stat.st_size:
                    skipped += 1
                    continue
        except Exception as e:
            print(f"Error stat-ing {clean_path}: {e}")
            pass
            
        print(f"Downloading -> {clean_path}")
        try:
            sftp.get(remote_full_path, local_path)
            downloaded += 1
        except Exception as e:
            print(f"Failed to download {clean_path}: {e}")
        
    sftp.close()
    client.close()
    print(f"\nSync complete! Downloaded: {downloaded}, Skipped (already up-to-date): {skipped}")

if __name__ == "__main__":
    pull_files()
