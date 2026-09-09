
#!/bin/bash
while [ ! -f ../../results/mu_EB_final.rds ]; do
  sleep 10
done
Rscript 04_factorial_final_parallel.R > factorial_final.log 2>&1
