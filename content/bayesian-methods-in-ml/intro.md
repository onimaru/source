(bayesian-methods-in-ml)=
# Bayesian Methods in Machine Learning

Here is a common situation in machine learning problems: we have our dataset, $\mathcal{D}$, and want, for example, a model with the ability to generate data as if they were real (usually conditional to something) or a model that works like the probality density function. There are many techniques to do such things and we will approach some here like:
- Energy-based models
- Variational autoencoder
- Generative adversarial networks

All of them begin with the same idea, the Bayes' theorem:

$$p(h \vert x) = \frac{p(h, x)}{p(x)} = \frac{p(x \vert h)p(h)}{p(x)},$$

where $x$ is the observed data ($x \sim p(x)$), $h$ is called parameter (it may be a latent or other observed variable) ($h \sim p(h)$) and the probability density functions $p(h)$, $p(x \vert h)$, $p(h \vert x)$ and $p(x)$ are respectively called prior, likelihood posterior and evidence. The random variables $x$ and $h$ can be discrete or continuous and for the later $p(x) = \int p(x, h)dh = \int p(x \vert h)p(h)dh$.

We need to focus in what is our goal here. There is a problem to solve and maybe we have some knowledge about the behavior of the data or the mechanism generating they. If that is the case, we can begin with some assumptions about what could be the density functions. However, it is very common that $p(x)$ is intractible, i.e., there is no closed form to compute the integral. In this case we could try to approximate the value of the integral.  
There are two main approaches to compute the approximation: Variational Inference (VI) and Markov Chain Monte Carlo (MCMC). Basically MCMC is asymptotically exact (law of large numbers), but is computationally expensive. VI on the other hand is faster, main option for large datasets, but there is no guarantee that it would lead to the correct approximation.

## Variational Inference

In VI, to find an approximation to $p(h \vert x)$ a candidate distribution, $q(h)$ is set to be this approximation. In order to find the candidate, an optimization framework is used, i.e., $q(h)$ will the one who minimizes some goal function with respect to $p(h \vert x)$. The main choice for the goal is Kullback-Liebler divergence (KL divergence):

$$q^{*}(h) = argmin_{q} KL[q(h)\vert \vert p(h \vert x)] = argmin_{q} \int q(h)\log{\left[\frac{q(h)}{p(h \vert x)} \right]}dh.$$

Rearranging the integral we have:

$$KL[q(h)\vert \vert p(h \vert x)] = \mathbb{E}_{q} \left[\log{q(h)}\right] - \mathbb{E}_{q}\left[\log{p(x,h)}\right] + \log{p(x)}.$$

Since $\log{p(x)}$ is independent of $q(h)$, it is a constant and therefore not needed during optimization. This is important because, as said early, $p(x)$ can be intractable, without it the problem is easier. The first two terms in the r.h.s. of the above equation has a special name, ELBO (Evidence Lower Bound),

$$ELBO(q) = \mathbb{E}_{q}\left[\log{p(x,h)}\right] - \mathbb{E}_{q} \left[\log{q(h)}\right]$$

Thus:

$$\log{p(x)} = ELBO(q) + KL[q(h)\vert \vert p(h \vert x)].$$

Remember that our solution is to use an approximation because it is difficult or impossible to compute $p(x)$. The KL divergence is algo intectable, but we can compute the $ELBO(q)$ term. Since the KL divergence is positive definite, maximizing (note that the signs were changed in the last equation) the $ELBO(q)$ only is a guarantee that $p(x)$ (or $\log{p(x)}$) is being maximized. 

That explains the name ELBO, it is a lower limit to the evidence, $p(x)$. If $KL[q(h)\vert \vert p(h \vert x)] = 0$, then the appoximation is exact. It is common to say that minimizing $KL[q(h)\vert \vert p(h \vert x)]$ is equivalent to maximizing the $ELBO(q)$.

Therefore, you can use this optimization approach to find a approximation for the $p(h \vert x)$, but this was just to define the objective function, the approximate $q(h)$ still must be chosen from a family of distributions (like, mean-field or exponential) as well as the optimization method.

The Variational Autoencoder (VAE) uses this approach with slightly modifications, instead of $q(h)$ it uses $q(h \vert x)$ as a neural network and the KL divergence is set as $KL[q(h \vert x)\vert \vert \mathcal{N}(0,1)]$. There are other differences like the use of the reparametrization trick, but will see more details in the appropriate section.

## MCMC


$$KL[q(h \vert x)\vert \vert p(h \vert x)] = \mathbb{E}_{q} \left[\log{q(h \vert x)}\right] - \mathbb{E}_{q}\left[\log{p(x,h)}\right] -\mathbb{E}_{q}\left[\log{p(h)}\right] + \log{p(x)}.$$