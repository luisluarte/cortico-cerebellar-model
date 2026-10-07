
$content = Get-Content C:\Users\DCCS5\.gemini\antigravity\brain\ee7b6b70-a0ae-4607-9cdf-f55667b1cc2c\post_mortem.md
$content += ""
$content += "### 3. The 'Illusion of Convergence' (The Z-Scale MCMC Bug)"
$content += "During the final verification of the Bayesian ELPD, I discovered a critical implementation error in how the NSGA-II parameters were handed off to the Custom Random-Walk MCMC. The MCMC sampler operates on an unconstrained Logit scale ($Z \in (-\infty, \infty)$) and uses `inv_logit_scaled` to map the proposals back into the strict biological boundaries."
$content += ""
$content += "However, I accidentally injected the empirical priors (`mu_EB`) using the raw parameter scale rather than transforming them via `qlogis()`. This caused the MCMC to map the already-scaled values through `inv_logit_scaled` a second time, pushing the parameters far outside their stable regimes."
$content += ""
$content += "**The Pathology:**"
$content += "- The wildly out-of-bounds parameters caused the Euler integration of the Cortical leaky accumulator to blow up to `NaN`."
$content += "- The C++ likelihood function caught the `NaN` and returned a hardcoded penalty of `-1e9`."
$content += "- Because **every single proposed step** blew up and returned `-1e9`, the Metropolis-Hastings acceptance ratio ($\alpha = \frac{L_{new}}{L_{old}}$) became $\frac{-1e9}{-1e9} = 1.0$."
$content += "- The sampler blindly accepted every step, essentially executing a pure random walk that perfectly sampled the `N(0,1)` prior."
$content += ""
$content += "As a result, standard MCMC diagnostics like $\hat{R} < 1.01$ and $Divergences = 0$ reported **perfect convergence**. It was only by strictly auditing the final pointwise Expected Log Predictive Density (ELPD) — which revealed an impossible score of `-50,000,000,000` — that the silent failure was caught."
$content += ""
$content += "I have since applied the `qlogis()` transformation to correctly map `mu_EB` into the $Z$-space, ensuring the MCMC navigates the actual posterior landscape."
Set-Content C:\Users\DCCS5\.gemini\antigravity\brain\ee7b6b70-a0ae-4607-9cdf-f55667b1cc2c\post_mortem.md -Value $content

