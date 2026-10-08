import re

with open('poster.tex', 'r') as f:
    content = f.read()

# Remove plot_memory_capacity.png from Left Column
mem_cap_block = r"\\vspace\{0\.5cm\}\s*\\begin\{figure\}.*?plot_memory_capacity\.png.*?\\end\{figure\}"
content_no_mem = re.sub(mem_cap_block, '', content, flags=re.DOTALL)

# Add it to Right Column after plot_encoding_ABC.png
enc_block = r"(\\begin\{figure\}.*?plot_encoding_ABC\.png.*?\\end\{figure\})"
replacement = r"\1\n        \\vspace{0.5cm}\n        \\begin{figure}\n            \\centering\n            \\includegraphics[width=\\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_memory_capacity.png}\n            \\caption{\\textbf{Temporal Integration.} The baseline $M_{cc}$ maintains temporal memory, while complete thalamic ablation collapses into a Markovian reactor.}\n        \\end{figure}"

content_new = re.sub(enc_block, replacement, content_no_mem, flags=re.DOTALL)

# Also change all heights to \linewidth
content_new = re.sub(r'height=10cm', r'width=\\linewidth', content_new)

with open('poster_test.tex', 'w') as f:
    f.write(content_new)
