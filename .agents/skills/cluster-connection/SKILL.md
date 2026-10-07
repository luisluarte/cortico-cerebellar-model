---
name: cluster-connection
description: Instructions for securely connecting to the 32-core Arch Linux remote cluster via SSH/Tailscale and executing parallel workloads.
---

# Arch-Compute Remote Cluster Connection Guide

This guide provides instructions for AI agents and users on how to connect to the Kotz/Luarte 32-core Arch Linux compute server. This server is strictly used for massive parallel workloads, such as NSGA-II hyperparameter searches, MCMC clustering, and high-performance `brms` modeling.

## 1. SSH Connection Aliases

The host machine is configured with SSH aliases located in `~/.ssh/config`. Authentication is handled automatically via the `~/.ssh/id_ed25519` key for the user `niconicoluarte`.

You can execute remote commands via PowerShell or bash using the following SSH commands:

### A. Remote Connection (Tailscale VPN)
When the local machine is off-site, connection is routed through the Tailscale mesh network.
- **Alias:** `arch-compute`
- **IP Address:** `100.121.224.32`
- **Command Example:** `ssh arch-compute "ls -la"`

### B. Local Connection (LAN)
When the local machine is on the same local area network as the server, use the LAN alias for lower latency.
- **Alias:** `arch-compute-lan`
- **IP Address:** `192.168.1.195`
- **Command Example:** `ssh arch-compute-lan "ls -la"`

## 2. Offloading Workloads

When you need to run heavy parallel workloads (e.g., array jobs, C++ compilations, R parallel clusters) that exceed local laptop memory/CPU limits:

1. **Upload Files:** Use `scp` to move necessary datasets or scripts to the remote server.
   *Example:* `scp -r src/ data/ arch-compute:~/cortico-cerebellar-model/`
2. **Execute:** SSH into the server and trigger the master script. 
   *Example:* `ssh arch-compute "cd ~/cortico-cerebellar-model && Rscript src/master_nsga2.R"`
3. **Parallel Configuration (R):** Ensure the remote R scripts are configured to utilize all 32 cores:
   ```R
   library(doParallel)
   cl <- makeCluster(32) # Using 32 cores for remote
   registerDoParallel(cl)
   ```
4. **Download Results:** Retrieve the generated outputs (`.rds`, `.csv`) back to the local machine using `scp`.
   *Example:* `scp arch-compute:~/cortico-cerebellar-model/results/* results/`

## 3. Remote Execution Guidelines for Agents
- **Do not poll aggressively:** If you start a long-running process on the remote server, detach the process (using `nohup` or `tmux`) rather than keeping the SSH connection blocking indefinitely, or use standard agent background task execution.
- **Hardware awareness:** The laptop host is memory constrained (3GB RAM). ALWAYS offload 32-core processing to `arch-compute`.
