| Page | Printed text | Correct text | Thanks |
|------|--------------|--------------|------|
| 1 | We can summarize the Bayesian modeling process using three steps | Missing citation of [Bayesian Data Analysis](https://sites.stat.columbia.edu/gelman/book) by Gelman et al | Bob Carpenter |
| 5 | So let’s take a walk through the garden of forking paths [Borges, 1944]. | In the context of statistics the "garden of forking paths" appears mentioned By [Andrew Gelman](http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf), Richard McElreath in his book [Statistical Rethinking](https://xcelab.net/rm/), see also this [wikipedia entry](https://en.wikipedia.org/wiki/Forking_paths_problem) | Bob Carpenter |
| 13 | The binomial coefficient is typeset as $\left(\frac{n}{x}\right)$ | it should be $\left(n\atop{}x\right)$ | Chris Hansen |
| 32 | There is a typo in the denominator of the normalizing constant | it should be $p(\theta) = \underbrace{\frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha) + \Gamma(\beta)}}_{\text{normalizing constant}} \;\; \theta^{\alpha-1} (1-\theta)^{\beta-1}$ | XIN Hongwei |
| 62 | We will explore the value **of** over a grid of 200 points. |  We will explore the value over a grid of 200 points.| dweights |
| 64 | ...and many operations applied to **Guassians** return another Gaussian. |  ...and many operations applied to **Gaussians** return another Gaussian. | marctagl65 |
| 88 (exercise 4) | wrong  short url  | The link should point to https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html#case-study-2-coal-mining-disasters  | DrEntropy |
| 94 | ...the variance between observed and theoretical values should be the same for all groups.| ...the variance between observed and theoretical values should be unique for each group. |  Kenji Oman  |
| 104 | Figure 3.6 indicates a HalfNormal |  Figure 3.6 should indicate a Gamma | Jacob Warren  |
| 115 | We are going to *usetemperature* | We are going to *use temperature* | Kenji Oman |
| 115 | The noise term is 𝜖 | The noise term is 𝜎 | Parrenin Frédéric |
| 116 | We **commit** it because otherwise... | We **omit** it because otherwise... | Kenji Oman |
| 122 | The variance of the NegativeBinomial is 𝜇 + 𝜇²/𝛼 , so the larger the value of 𝛼 the **larger** the variance. | The variance of the NegativeBinomial is 𝜇 + 𝜇²/𝛼 , so the larger the value of 𝛼 the **smaller** the variance.     |  Tomás Capretto  |
| 124 | (or data with a few bulk points)  | (or data with **only** a few bulk points) | Kenji Oman |
| 133 | We have been using the linear motif to model the mean of a distribution **and, in the previous section, we used it to model interactions.** In statistics,... | We have been using the linear motif to model the mean of a distribution.  In statistics,...  | Jacob Warren |
| 145 | In **the next chapter**, we will learn more about linear regression... | In **Chapter 6**, we will learn more about linear regression... | Tomás Capretto |
| 155 | In the equation for a polynomial model of order 5 all coefficient are the same ($\beta_{0}$) | it should be $\alpha + \beta_0 x + \beta_1 x^2 + \beta_2 x^3 + \beta_3 x^4 + \beta_4 x^5$ | Jarvin Jeffrey Gallego |
| 191 |  The utility of **plot_cap** ... | The utility of **plot_predictions**... | Tomás Capretto  |
| 194 | model_poly4 = bmb.Model("rented ∼ poly(**temperature**, degree=4)", bikes, | model_poly4 = bmb.Model("rented ∼ poly(**hour**, degree=4)", bikes, | Jacob Warren |
| 195 | Figure 6.5: Posterior mean and posterior predictive distribution for the **bikes model with temperature and humidity** | Figure 6.5: Posterior mean and posterior predictive distribution for the **polynomial bikes models with hour**. | Jacob Warren |
| 207 | We have been using **bmb.interpret_plot_predictions** ... One of them is **bmb.interpret_plot_comparisons**. | We have been using **bmb.interpret.plot_predictions** ... One of them is **bmb.interpret.plot_comparisons**.|  Tomás Capretto  |
| 208 | Another useful function is **bmb.interpret_plot_slopes** | Another useful function is **bmb.interpret.plot_slopes**  |  Tomás Capretto |
| 254 | We call **𝜙** the inverse link function and 𝜙 is... | We call **𝜓** the inverse link function and 𝜙 is... | Jacob Warren |
| 344 | **https://arviz-devs.github.io/Exploratory-Analysis-of-Bayesian-Models/** | **https://arviz-devs.github.io/EABM** |    |

Notes:

* On page 23, Figure 1.9 shows the kurtosis. What PreliZ actually computes is the "excess kurtosis", i.e the kurtosis -3. Thanks to Narinder Singh for pointing this out.

* On Code block 2.26 we use a Normal likelihood and in Code block 2.27 we use a Gamma likelihood. The text has been updated to reflect why we do the change (The Gamma likelihood is more appropriate for the data we are modeling, given that all values are positive). Thanks to Kenji Oman for pointing this out.