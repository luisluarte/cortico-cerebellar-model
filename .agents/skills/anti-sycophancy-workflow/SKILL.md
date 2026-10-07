---
name: Anti-Sycophancy Workflow
description: A strictly enforced two-step pipeline for all data analysis and statistical tasks to prevent hallucination and expectation bias.
---

# Anti-Sycophancy Workflow

When analyzing data or running statistical models, you are vulnerable to expectation bias—anticipating a result based on theory and hallucinating it before the data confirms it. 

To prevent this, you **MUST** follow this two-step pipeline unconditionally:

## Step 1: Raw Extraction (No Plotting)
1. Write the R/Python script to compute the statistics.
2. The script must NOT generate any plots (no ggplot2, no ggsave).
3. The script MUST output the raw numeric results to a .txt file (e.g., stats_summary.txt) or print them clearly to stdout.
4. Run the script and wait for it to finish.
5. If saved to a file, run a cat command to read the exact raw numbers into your context.

## Step 2: Verification and Plotting
1. You must read the raw numbers.
2. **MANDATORY**: Send the raw numbers to the Epistemic Auditor subagent using send_message. Wait for its reply.
3. Only AFTER the subagent confirms the direction of the effect are you allowed to:
   - Write a new script (or append to the old one) to generate the visualization.
   - Generate the markdown artifact.
   - Reply to the user with the final conclusion.

**CRITICAL**: Never combine Step 1 and Step 2 in a single turn. Execution must be split across two distinct phases to guarantee the numbers enter your context window before the narrative is formed.
