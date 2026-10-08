import sys

with open("poster.tex", "r") as f:
    lines = f.readlines()

def get_block(start_line_idx):
    idx = start_line_idx
    block_lines = []
    while idx < len(lines):
        block_lines.append(lines[idx])
        if "\\end{block}" in lines[idx]:
            return block_lines, idx
        idx += 1
    return block_lines, idx

# Find Introduction
intro_start = -1
for i, l in enumerate(lines):
    if "\\begin{block}{Introduction}" in l:
        intro_start = i
        break

intro_block, intro_end = get_block(intro_start)

# Find Methods
methods_start = -1
for i, l in enumerate(lines):
    if "\\begin{block}{Methods}" in l:
        methods_start = i
        break

methods_block, methods_end = get_block(methods_start)

# Find Results
results_start = -1
for i, l in enumerate(lines):
    if "\\begin{block}{Results}" in l:
        results_start = i
        break

results_block, results_end = get_block(results_start)

# Find Results (cont.)
results_cont_start = -1
for i, l in enumerate(lines):
    if "\\begin{block}{Results (cont.)}" in l:
        results_cont_start = i
        break

results_cont_block, results_cont_end = get_block(results_cont_start)

# Combine results
combined_results = ["      \\begin{block}{Results}\n"] + results_block[1:-1] + results_cont_block[1:-1] + ["      \\end{block}\n"]

# Now reconstruct the body
# Everything before Intro
pre_intro = lines[:intro_start]

# After intro, add Methods
left_col_after_intro = methods_block

# Find column split
# The left column ends at "\end{column}" right before right column
split_start = -1
for i in range(intro_end, len(lines)):
    if "% RIGHT COLUMN" in lines[i]:
        split_start = i - 1 # include the \end{column}
        break

# The right column starts at split_start
middle = ["    \\end{column}\n", "    \n", "    % RIGHT COLUMN\n", "    \\begin{column}{.45\\textwidth}\n\n"]

# We need to get Conclusion and Acks
conc_start = -1
for i, l in enumerate(lines):
    if "\\begin{block}{Conclusion}" in l:
        conc_start = i
        break
conc_block, conc_end = get_block(conc_start)

ack_start = -1
for i, l in enumerate(lines):
    if "\\begin{block}{Acknowledgements" in l:
        ack_start = i
        break
ack_block, ack_end = get_block(ack_start)

post_ack = lines[ack_end+1:]

new_lines = pre_intro + intro_block + ["\n"] + methods_block + ["\n"] + middle + combined_results + ["\n"] + conc_block + ["\n"] + ack_block + post_ack

with open("poster_new.tex", "w") as f:
    f.writelines(new_lines)
