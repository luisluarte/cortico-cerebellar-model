with open('poster.tex', 'r') as f:
    content = f.read()

import re
# extract everything up to begin{block}{Results}
before_results = content[:content.find(r'      \begin{block}{Results}')]
# extract everything after end{block} for results
after_results = content[content.find(r'      \end{block}', content.find(r'      \begin{block}{Results}')) + 18:]

results_left = r"""      \begin{block}{Results}
        \vspace{0.5cm}
        \begin{figure}
            \centering
            \includegraphics[width=0.85\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_nsga2_hv.png}
            \caption{\textbf{NSGA-II Discovery.} Multi-objective evolution jointly minimized distillation and empirical loss to converge on structural bounds.}
        \end{figure}
        
        \vspace{0.8cm}
        \begin{figure}
            \centering
            \includegraphics[width=0.85\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_psis_loo.png}
            \caption{\textbf{Zero-Shot Transfer.} The distilled physiological model predicts raw behavior significantly better than natively optimized heuristics.}
        \end{figure}
      \end{block}"""

content = before_results + results_left + after_results

results_right = r"""      \begin{block}{Results (cont.)}
        \vspace{0.5cm}
        \begin{figure}
            \centering
            \includegraphics[width=0.85\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_memory_capacity.png}
            \caption{\textbf{Temporal Integration.} The baseline $M_{cc}$ maintains temporal memory, while complete thalamic ablation collapses into a Markovian reactor.}
        \end{figure}
        
        \vspace{0.8cm}
        \begin{figure}
            \centering
            \includegraphics[width=0.85\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_encoding_ABC.png}
            \caption{\textbf{Directional Flow.} Granger causality reveals significant causal flow from Cerebellum to Cortex across Value, State, and Volatility.}
        \end{figure}
      \end{block}

      \begin{block}{Methods}"""

content = content.replace(r'      \begin{block}{Methods}', results_right)

with open('poster_test.tex', 'w') as f:
    f.write(content)
