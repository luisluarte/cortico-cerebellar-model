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

def find_block(name):
    for i, l in enumerate(lines):
        if f"\\begin{{block}}{{{name}}}" in l:
            return get_block(i)
    return [], -1

intro_block, _ = find_block("Introduction")
methods_block, _ = find_block("Methods")
conc_block, _ = find_block("Conclusion")
ack_block, _ = find_block("Acknowledgements")
res1_block, _ = find_block("Results")
res2_block, _ = find_block("Results (cont.)")

# Combine results content to extract all figures
res_content = res1_block[1:-1] + res2_block[1:-1]
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
# Left column: Intro, Methods, Results (Text + Fig1)
left_res = ["      \\begin{block}{Results}\n"]
for f in figures[:2]:  # Text is [0], Fig1 is [1]
    left_res.extend(f)
    left_res.append("        \\vspace{0.1cm}\n")
left_res.append("      \\end{block}\n")

# Right column: Results (Fig2, Fig3, Fig4, Fig5), Conclusion, Acks
right_res = ["      \\begin{block}{Results (cont.)}\n"]
for f in figures[2:]:
    right_res.extend(f)
    right_res.append("        \\vspace{0.1cm}\n")
right_res.append("      \\end{block}\n")

prefix = []
for l in lines:
    if "\\begin{block}" in l:
        break
    prefix.append(l)

middle = ["    \\end{column}\n", "    \n", "    % RIGHT COLUMN\n", "    \\begin{column}{.45\\textwidth}\n\n"]
suffix = ["    \\end{column}\n", "  \\end{columns}\n", "\\end{frame}\n", "\\end{document}\n"]

new_lines = prefix + intro_block + ["\n"] + methods_block + ["\n"] + left_res + ["\n"] + middle + right_res + ["\n"] + conc_block + ["\n"] + ack_block + ["\n"] + suffix

# Fix widths
for i in range(len(new_lines)):
    if "includegraphics" in new_lines[i] and "plot" in new_lines[i]:
        new_lines[i] = "            \\includegraphics[width=0.6\\linewidth,keepaspectratio]{" + new_lines[i].split("{")[1]

with open("poster_new.tex", "w") as f:
    f.writelines(new_lines)
