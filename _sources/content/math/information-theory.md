# Information Theory
---

Suppose we have a random variable $x$ and we are interested in how much information we get when $x$ is measured. If the event is highly improbable we have more information than if it is highly probable. Information is given by a quantity $h(x)$ that is a function of the probability distribution $p(x)$. If two events are observed and unrelated the total information will be the sum of information in both events:

$$h(x,y) = h(x) + h(y)\ \iff \ p(x,y) = p(x)p(y)$$

We see that information depends on the logarithm of the probability distribution. For a discrete $x$ we have

$$h(x) = - log_{2}p(x).$$

The expectation of this quantity is what we call `entropy` of the variable $x$,

$$H[x] = -  \sum_{x} p(x) log_{2}p(x)$$

If there's no chance o observing $x$, i.e. $p(x)=0$, then we get no information $H[x]=0$.

As an example, suppose we observe the throw of a fair 8-sided dice and we want to transmit the information about the expectation:
```python
import numpy as np

H = -8*(1/8)*np.log2(1/8)
H
>>> 3.0
```
This is measured in `bits`. In this case we need at least a 3 `bits` number to transmit the information, no less than that. When $x$ is continuous it is common to use the natural logarithm and the unit becomes `nats` instead of bits.

Let us see an example of three distributions with the same mean but different standard deviation.

```python
p1 = np.array([0.2,0.2,0.2,0.2,0.2])
p2 = np.array([0.1,0.15,0.5,0.15,0.1])
p3 = np.array([0.0,0.005,0.99,0.005,0.0])
```
Define a function to compute entropy:
```python
def H(p):
    return np.sum([-val*np.log(val) if val > 0 else 0 for val in p])

H(p1),H(p2),H(p3)
(1.6094, 1.3762, 0.0629)
```

The narrower distribution has a smaller entropy. This is because the uncertainty about its expectation is smaller. If some $ p(x_{i})=1 $ all other $p(x_{j \ne i})=0$, uncertainty is zero and we get no information $H(x_{i})=0$.

## Differential Entropy

We can use a continuous distribution, $p(x)$ and the discrete sum becomes an integral and the entropy is usually called `differential entropy`:  

$$H[x] = -\int p(x) \ln p(x) dx.$$

We can now use `Lagrange multipliers` to find a distribution which maximizes the differential entropy. To do that we need to constrain the first and second moments of $p(x)$:

$$\int_{-\infty}^{+\infty}p(x)dx = 1,$$  

$$\int_{-\infty}^{+\infty} x p(x)dx = \mu,$$  

$$\int_{-\infty}^{+\infty} (x- \mu)^{2} p(x)dx = \sigma^{2}.$$  

Setting the `variational derivative` to zero we get as solution $p(x) = \exp{(-1 + \lambda_{1} + \lambda_{2} x + \lambda_{3} (x - \mu)^{2})}$. Using substitution we get:

$p(x) = \frac{1}{\sqrt{2 \pi \sigma^{2}}}\exp{\left(-\frac{(x - \mu)^{2}}{2 \sigma^{2}}\right)}$

The Gaussian distribution is the one that maximizes differential entropy which is:

$$H[x] =  -\int_{-\infty}^{+\infty} \mathcal{N}(x\vert \mu, \sigma^{2}) \ln \mathcal{N}(x\vert \mu, \sigma^{2}) dx = \frac{1}{2}(1 + \ln{2 \pi \sigma^{2}})$$

This result agrees with what we found earlier, the entropy increases as the distribution becomes broader, i.e., $\sigma^{2}$ increases.

## Conditional Entropy

Suppose now we have a joint distribution $p(x,y)$ and have an observation of $x$. It is possible to compute the additional information needed to specify the corresponding observation of $y$ with $- \ln p(y\vert x).$ Thus the average additional information to specify $y$, called `conditional entropy` of $y$ given $x$, is:

$$H[y\vert x] = -\int \int p(y,x) \ln{ p(y\vert x)}dydx$$

Using $p(y\vert x) = \frac{p(y,x)}{p(x)}$ we get:

$$H[y\vert x] = H[y,x] - H[x].$$

## Kullback-Leibler Divergence

Suppose we have an unknown distribution $p(x)$ and we are using another distribution $q(x)$ to model it. We can compute the additional amount of information needed to specify $x \sim p(x)$ when we observe $x \sim q(x)$:

$$KL(p\vert \vert q) = -\int p(x) \ln q(x) dx + \int p(x) \ln p(x) dx = -\int p(x) \ln{\left(\frac{q(x)}{p(x)} \right)}dx.$$

This `relative entropy` is called `Kullback-Leibler Divergence` or simply `KL Divergence` and is a measure of dissimilarity between two distributions. Note that it is anti-symmetric,

$$KL(p\vert \vert q) \ne KL(q\vert \vert p).$$

To accomplish the task of approximating $q(x)$ to $p(x)$ we may observe $x \sim p(x)$ a finite number of times, $N$, use $q$ as a parametric function, $q(x \vert \theta)$ and use the expectation of $KL(p \vert \vert q).$

For a function $f(x)$ its expectation is $\mathbb{E}[f] = \int p(x) f(x)dx$ and for a $N$ number of observations it becomes a finite sum:

$$\mathbb{E}[f] \approx \frac{1}{N} \sum_{n=1}^{N} f(x_{n}).$$

For the KL divergence we have:

$$KL(p \vert \vert q) \approx \frac{1}{N} \sum_{n=1}^{N}(-\ln q(x_{n} \vert \theta) + \ln p(x_{n})).$$

The first therm is the `negative log likelihood` for $\theta$ under distribution $q(x\vert \theta)$. Because of that people usually say that `minimizing the KL divergence is equivalent to maximizing the likelihood function`.

As an exercise we can find $KL(p\vert \vert q)$ where $p(x)=\mathcal{N}(x\vert \mu, \sigma^{2})$ and $q(x)=\mathcal{N}(x\vert m, s^{2})$.

$KL(p\vert \vert q) = -\int p(x) \ln{\frac{q(x)}{p(x)}}dx$  
$KL(p\vert \vert q) = -\int p(x) \ln{q(x)}dx + \int p(x) \ln{p(x)}dx$  
$KL(p\vert \vert q) = \frac{1}{2}\int p(x) \ln{2 \pi s^{2}}dx + \frac{1}{2s^{2}}\int p(x)(x-m)^{2}dx - \frac{1}{2}(1+\ln{2 \pi \sigma^{2}})$  
$KL(p\vert \vert q) = \frac{1}{2}\ln{2 \pi s^{2}} - \frac{1}{2}(1+\ln{2 \pi \sigma^{2}})+ \frac{1}{2s^{2}}(\langle x \rangle^{2} -2m \langle x \rangle + m^{2})$  
$KL(p\vert \vert q) = \frac{1}{2} \ln{\frac{s^{2}}{\sigma^{2}}} - \frac{1}{2} + \frac{\sigma^{2}+(\mu - m)^{2}}{2s^{2}}$  
$KL(p\vert \vert q) = \ln{\frac{s}{\sigma}} - \frac{1}{2} + \frac{\sigma^{2}+(\mu - m)^{2}}{2s^{2}},$  
where $(\langle x \rangle -m)^{2} -\langle x \rangle^{2} = -2 \langle x \rangle m + m^{2}$.

### f-Divergence

The KL divergence is the most famous function of a broader family called `f-Divergences`, more generally defined as:

$D_{f}(p \vert \vert q) \equiv \frac{4}{1 - f^{2}} \left( 1 - \int p(x)^{\left(\frac{1 + f}{2} \right)} q(x)^{\left(\frac{1 - f}{2} \right)}dx \right)$

where $f$ is a continuous real parameter, $-\infty \le f \le + \infty$. Some special cases are:

$KL(p\vert \vert q) = \lim_{f \rightarrow 1} D_{f}(p \vert \vert q),$

$KL(q\vert \vert p) = \lim_{f \rightarrow -1} D_{f}(p \vert \vert q)$

and the `Hellinger distance`

$D_{H}(p \vert \vert q) = \lim_{f \rightarrow 0}  D_{f}(p \vert \vert q) = \int \left( p(x)^{2} - q(x)^{2} \right)^{2}dx.$

Since we only work with distributions with compact support $D_{f}(p \vert \vert q) \ge 0$, again it is zero if and only if $p(x) = q(x)$.

## Mutual Information

For a joint distribution $p(y,x)$ the KL divergence can be used to quantify how close the two variables are to be independent, i.e., $p(y,x) = p(y)p(x)$, this is called `mutual information` between $x$ and $y$:

$I[y,x] = KL(p(y,x)\vert \vert p(y)p(x)) = -\int \int p(y,x) \ln \frac{p(y)p(x)}{p(y,x)}dydx$.  

$I[y,x] \ge 0$ and $I[y,x] = 0 \iff$ $x$ and $y$ are independent.

Mutual information can be written with conditional entropy:

$I[y,x] = H[y] - H[y\vert x] = H[x] - H[x\vert y]$.
