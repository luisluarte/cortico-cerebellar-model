import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.121.224.32', username='niconicoluarte', password='dw4yb26f')
sftp = client.open_sftp()
sftp.get('cortico-cerebellar-model/results/brms_switch_summary_spline_global.txt', 'results/brms_switch_summary_spline_global.txt')
sftp.get('cortico-cerebellar-model/results/brms_switch_model_spline_global.rds', 'results/brms_switch_model_spline_global.rds')
client.close()
