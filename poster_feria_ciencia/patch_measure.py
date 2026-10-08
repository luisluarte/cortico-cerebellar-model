with open('poster_test.tex', 'r') as f:
    content = f.read()

# remove figures from right column
content = content.replace(r"""        \begin{figure}
            \centering
            \includegraphics[width=0.85\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_memory_capacity.png}
            \caption{\textbf{Temporal Integration.} The baseline $M_{cc}$ maintains temporal memory, while complete thalamic ablation collapses into a Markovian reactor.}
        \end{figure}""", "")
content = content.replace(r"""        \begin{figure}
            \centering
            \includegraphics[width=0.85\linewidth,keepaspectratio]{../cortico-cerebellar-model/presentations/plots/plot_encoding_ABC.png}
            \caption{\textbf{Directional Flow.} Granger causality reveals significant causal flow from Cerebellum to Cortex across Value, State, and Volatility.}
        \end{figure}""", "")

with open('poster_measure.tex', 'w') as f:
    f.write(content)
