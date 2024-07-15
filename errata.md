| Page | Printed text | Correct text | Thanks |
|------|--------------|--------------|------|
|  88 (exercise 4) | wrong  short url  | The link should point to https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html#case-study-2-coal-mining-disasters  | DrEntropy |
| 104 | Figure 3.6 indicates a HalfNormal |  Figure 3.6 should indicate a Gamma | Jacob Warren  |
| 115 | The noise term is 𝜖 | The noise term is 𝜎 | Parrenin Frédéric |
| 122  | The variance of the NegativeBinomial is 𝜇 + 𝜇²/𝛼 , so the larger the value of 𝛼 the **larger** the variance. | The variance of the NegativeBinomial is 𝜇 + 𝜇²/𝛼 , so the larger the value of 𝛼 the **smaller** the variance.     |  Tomás Capretto  |
| 133 | We have been using the linear motif to model the mean of a distribution **and, in the previous section, we used it to model interactions.** In statistics,... | We have been using the linear motif to model the mean of a distribution.  In statistics,...  | Jacob Warren |
| 145  | In **the next chapter**, we will learn more about linear regression... | In **Chapter 6**, we will learn more about linear regression... | Tomás Capretto |
|  191 |  The utility of **plot_cap** ... | The utility of **plot_predictions**... | Tomás Capretto  |
| 194 | model_poly4 = bmb.Model("rented ∼ poly(**temperature**, degree=4)", bikes, | model_poly4 = bmb.Model("rented ∼ poly(**hour**, degree=4)", bikes, | Jacob Warren |
| 195 | Figure 6.5: Posterior mean and posterior predictive distribution for the bikes model with temperature and humidity | Figure 6.5: Posterior mean and posterior predictive distribution for the polynomial bikes models with hour. | Jacob Warren |
|  207 | We have been using **bmb.interpret_plot_predictions** ... One of them is **bmb.interpret_plot_comparisons**. | We have been using **bmb.interpret.plot_predictions** ... One of them is **bmb.interpret.plot_comparisons**.|  Tomás Capretto  |
| 208  | Another useful function is **bmb.interpret_plot_slopes** | Another useful function is **bmb.interpret.plot_slopes**  |  Tomás Capretto |