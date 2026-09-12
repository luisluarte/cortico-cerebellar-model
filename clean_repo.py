import os
import glob
import shutil

# Target directory
root_dir = r"C:\Users\DCCS5\Documents\GitHub\cortico-cerebellar-model"
os.chdir(root_dir)

# 1. Manuscript Clutter
clutter_exts = ['*.aux', '*.bbl', '*.bcf', '*.blg', '*.log', '*.run.xml', '*.synctex.gz', '*.out', '*.snm', '*.nav', '*.toc']
for folder in ['manuscript', 'presentations']:
    for ext in clutter_exts:
        for f in glob.glob(os.path.join(folder, ext)):
            print(f"Deleting manuscript clutter: {f}")
            os.remove(f)

# Also delete any .log in the root and subdirectories (excluding .git)
for root, dirs, files in os.walk('.'):
    if '.git' in root or '.agents' in root or '.gemini' in root:
        continue
    for file in files:
        if file.endswith('.log') or file.endswith('_pid.txt'):
            fpath = os.path.join(root, file)
            print(f"Deleting scrap log: {fpath}")
            os.remove(fpath)

# 2. Legacy Stan Models and Scripts
stan_files = glob.glob('**/*.stan', recursive=True)
for f in stan_files:
    print(f"Deleting Stan model: {f}")
    os.remove(f)

# Binary compilations of stan files (often extensionless in Unix, but on Windows they might be .exe or extensionless)
# We know some names from the audit: wsls, readout, q_learning_cf
binaries = ['src/r/wsls', 'src/r/wsls.exe', 'src/r/readout', 'src/r/readout.exe', 
            'src/r/readout_intercept', 'src/r/readout_intercept.exe', 'src/r/readout_pooled', 'src/r/readout_pooled.exe',
            'src/r/wsls_fixed', 'src/r/wsls_fixed.exe', 'src/r/wsls_pooled', 'src/r/wsls_pooled.exe',
            'src/r/wsls_posteriors', 'src/r/wsls_posteriors.exe',
            'cerebellar_reservoir_ddm_model/q_learning_cf', 'cerebellar_reservoir_ddm_model/q_learning_cf.exe']
for b in binaries:
    if os.path.exists(b):
        print(f"Deleting Stan binary: {b}")
        os.remove(b)

# Legacy runner scripts
legacy_patterns = ['eval_m*.R', 'run_m*.R', 'run_d*.R', 'm1_*.stan', 'eval_wsls*.R', 'run_benchmark*.R', 'run_n10*.R', 'run_stan_bio_baseline*.R']
for pattern in legacy_patterns:
    for f in glob.glob(f"**/{pattern}", recursive=True):
        print(f"Deleting legacy runner: {f}")
        os.remove(f)

# 3. Scrap & Temporary logs / test scripts
scrap_patterns = ['nsga2_checkpoint*.csv', 'nsga2_checkpoint*.rds', 'pareto_front*.csv', 'search_beta_metrics*.R', 'test_deviation.R', 'test_qlearning.R', 'test_empirical_beta.R', 'test_torch.R', 'check_ids.R', 'check_trials.R', 'inspect_*.R', 'search_landscape.txt', 'all_user_inputs.txt', 'digested*.txt']
for pattern in scrap_patterns:
    for f in glob.glob(f"**/{pattern}", recursive=True):
        print(f"Deleting scrap file: {f}")
        os.remove(f)

# 4. Old Test Beds
if os.path.exists('reservoir_model'):
    print("Deleting reservoir_model directory...")
    shutil.rmtree('reservoir_model')

# cerebellar_reservoir_ddm_model has lots of scrap, but let's carefully keep data if any (actually we have it in root data/)
if os.path.exists('cerebellar_reservoir_ddm_model'):
    print("Deleting cerebellar_reservoir_ddm_model directory...")
    shutil.rmtree('cerebellar_reservoir_ddm_model')

print("Cleanup complete.")
