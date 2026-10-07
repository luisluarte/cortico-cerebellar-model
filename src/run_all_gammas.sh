
#!/bin/bash
echo "Starting Gamma Array Sweep on 32 Cores!"
for g in 0.0 0.25 0.50 0.75 1.0
do
  echo "Running gamma = $g..."
  Rscript master_gamma_array_mod.R $g
done
echo "All NSGA-II runs completed!"

