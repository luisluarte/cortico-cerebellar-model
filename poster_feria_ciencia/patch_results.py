import re

with open('poster.tex', 'r') as f:
    content = f.read()

headline = r"""    \begin{column}{.6\textwidth}
      \centering
      {\huge \textbf{\inserttitle} \\[1.5ex]}
      {\large \insertauthor \\[1ex]}
      {\normalsize \insertinstitute}
    \end{column}"""

content = re.sub(r'    \\begin{column}{.6\\textwidth}.*?    \\end{column}', headline, content, flags=re.DOTALL)

results_block = r"""      \begin{block}{Results}
        \vspace{0.5cm}
        \begin{figure}
            \begin{minipage}[t]{0.48\linewidth}
                \centering
                \includegraphics[width=\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_nsga2_hv.png}
                \caption{\textbf{NSGA-II Discovery.} Multi-objective evolution jointly minimized distillation and empirical loss to converge on structural bounds.}
            \end{minipage}\hfill
            \begin{minipage}[t]{0.48\linewidth}
                \centering
                \includegraphics[width=\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_psis_loo.png}
                \caption{\textbf{Zero-Shot Transfer.} The distilled physiological model predicts raw behavior significantly better than natively optimized heuristics.}
            \end{minipage}
        \end{figure}
        
        \vspace{0.5cm}
        \begin{figure}
            \begin{minipage}[t]{0.48\linewidth}
                \centering
                \includegraphics[width=\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_memory_capacity.png}
                \caption{\textbf{Temporal Integration.} The baseline $M_{cc}$ maintains temporal memory, while complete thalamic ablation collapses into a Markovian reactor.}
            \end{minipage}\hfill
            \begin{minipage}[t]{0.48\linewidth}
                \centering
                \includegraphics[width=\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_encoding_ABC.png}
                \caption{\textbf{Directional Flow.} Granger causality reveals significant causal flow from Cerebellum to Cortex across Value, State, and Volatility.}
            \end{minipage}
        \end{figure}
      \end{block}"""

content = re.sub(r'      \\begin{block}{Results}.*?      \\end{block}', results_block, content, flags=re.DOTALL)

with open('poster.tex', 'w') as f:
    f.write(content)
