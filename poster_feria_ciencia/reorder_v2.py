import sys

with open("poster.tex", "r") as f:
    lines = f.readlines()

def get_block(start_line_idx):
    if start_line_idx == -1: return [], -1
    idx = start_line_idx
    block_lines = []
    while idx < len(lines):
        block_lines.append(lines[idx])
        if "\\end{block}" in lines[idx]:
            return block_lines, idx
        idx += 1
    return block_lines, idx

# Blocks are: Intro, Methods, Results, Conclusion, Acks
def find_block(name):
    for i, l in enumerate(lines):
        if f"\\begin{{block}}{{{name}}}" in l:
            return get_block(i)
    return [], -1

intro_block, _ = find_block("Introduction")
methods_block, _ = find_block("Methods")
conc_block, _ = find_block("Conclusion")
ack_block, _ = find_block("Acknowledgements")

# Results needs to be split
results_block, _ = find_block("Results")

# The results block has 5 figures + 1 text \begin{figure}
# Let's extract the figures
res_content = results_block[1:-1]
figures = []
current_fig = []
in_fig = False
for line in res_content:
    if "\\begin{figure}" in line:
        in_fig = True
        current_fig = [line]
    elif "\\end{figure}" in line:
        current_fig.append(line)
        figures.append(current_fig)
        in_fig = False
    elif in_fig:
        current_fig.append(line)

# Let's rebuild the columns
# Left column: Intro, Methods, Results (Text + Fig1 + Fig2)
# Text is figures[0], Fig1 is figures[1], Fig2 is figures[2]
left_res = ["      \\begin{block}{Results}\n"]
for f in figures[:3]:
    left_res.extend(f)
    left_res.append("        \\vspace{0.1cm}\n")
left_res.append("      \\end{block}\n")

# Right column: Results (Fig3 + Fig4 + Fig5), Conclusion, Acks
right_res = ["      \\begin{block}{Results (cont.)}\n"]
for f in figures[3:]:
    right_res.extend(f)
    right_res.append("        \\vspace{0.1cm}\n")
right_res.append("      \\end{block}\n")

pre_intro = lines[:intro_block_start] if 'intro_block_start' in locals() else lines[:83] 
# Actually just hardcode the prefix
prefix = []
for l in lines:
    if "\\begin{block}" in l:
        break
    prefix.append(l)

middle = ["    \\end{column}\n", "    \n", "    % RIGHT COLUMN\n", "    \\begin{column}{.45\\textwidth}\n\n"]
suffix = ["    \\end{column}\n", "  \\end{columns}\n", "\\end{frame}\n", "\\end{document}\n"]

new_lines = prefix + intro_block + ["\n"] + methods_block + ["\n"] + left_res + ["\n"] + middle + right_res + ["\n"] + conc_block + ["\n"] + ack_block + ["\n"] + suffix

with open("poster_new.tex", "w") as f:
    f.writelines(new_lines)

