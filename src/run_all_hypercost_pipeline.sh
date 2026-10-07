#!/bin/bash
cd ~/cortico-cerebellar-model
echo "Running 4D projection..."
Rscript src/master_4d_hypercost.R
echo "Bootstrapping slopes..."
Rscript src/bootstrap_slopes_hypercost.R
