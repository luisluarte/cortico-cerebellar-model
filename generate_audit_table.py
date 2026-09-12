import os
import re

def guess_importance(filename, path):
    core_files = [
        "sim_bio_probes.cpp", "sim_hallucination.cpp", "sim_agent.cpp", "sim_restless_agent.cpp",
        "distillation_targets_v2.rds", "behavioral_compilate.csv", "factorial_fixed_10k_A.rds",
        "nsga2_pruning_results.rds", "decode_latent_traces.R", "correlate_representation.R",
        "run_hallucination_beta_thal.R", "run_lesion_beta_thal.R", "measure_confidence.R",
        "plot_behavioral_landscape.R", "plot_binned_markovian.R", "run_agent_simulation.R",
        "run_restless_sweep.R", "test_params.R", "01_preprocess.R", "02_rnn_oracle.R",
        "03_nsga2_final.R", "11_factorial_fixed_10k.R", "participant_characterization.R",
        "markovian_index_analysis.R"
    ]
    
    if filename in core_files or filename.startswith("bio_mcmc_") or filename.startswith("Fig"):
        return "Core: Part of the project"
    
    legacy_patterns = [
        r"wsls.*", r"readout.*", r"run_n10.*", r"run_benchmark.*", r"^m\d_.*", r"^d\d_.*", r"^eval_m\d.*", r"^run_m\d.*"
    ]
    for p in legacy_patterns:
        if re.match(p, filename):
            return "Legacy: Unused/Baseline model"
            
    scrap_patterns = [
        r"test_.*\.R", r"inspect_.*\.R", r"check_.*\.R", r"search_beta.*\.R", r"scrap.*", r"test_torch.*"
    ]
    for p in scrap_patterns:
        if re.match(p, filename) and filename != "test_params.R":
            return "Scrap: Just a script to test things"
            
    if filename.endswith(".log") or filename.endswith(".txt") or filename.endswith(".csv"):
        if filename == "behavioral_compilate.csv":
            return "Core Data"
        return "Log/Data: Likely scrap or intermediate"
        
    if "nsga2" in filename and filename not in core_files:
        return "Legacy: Intermediate Optimization"
        
    if "final_model" in path:
        return "Core: Generative Pipeline component"
        
    return "Unknown: Needs manual review"

def get_description(filepath):
    try:
        if filepath.endswith(('.png', '.pdf')):
            return "Figure / Plot"
        if filepath.endswith(('.rds', '.pt')):
            return "Compiled Model / Dataset"
        if filepath.endswith(('.csv', '.json', '.txt', '.log')):
            return "Data / Log File"
            
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline().strip() for _ in range(5)]
            text = " ".join([l for l in lines if l])
            if len(text) > 50:
                return text[:50] + "..."
            elif len(text) > 0:
                return text
            else:
                return "Empty or no header"
    except Exception as e:
        return f"Could not read"

def main():
    root_dir = r"C:\Users\DCCS5\Documents\GitHub\cortico-cerebellar-model"
    out_file = r"C:\Users\DCCS5\.gemini\antigravity\brain\ee7b6b70-a0ae-4607-9cdf-f55667b1cc2c\repo_audit_table.md"
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("# Repository Audit Table\n\n")
        f.write("| File Name | Relative Path | Derived Function | Importance |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude unwanted dirs
            dirnames[:] = [d for d in dirnames if d not in ['.git', '.agents', '.gemini']]
            
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir).replace('\\', '/')
                
                desc = get_description(full_path)
                imp = guess_importance(filename, rel_path)
                
                f.write(f"| `{filename}` | `{rel_path}` | {desc} | {imp} |\n")
                
    print(f"Table successfully written to {out_file}")

if __name__ == '__main__':
    main()
