# Laplace Approximation

This is a framework to find an Gaussian approximation to a probability density. Since it is Gaussian it is defined over continuous variables.

**Summary**: it is actually really simple. We need to find de `mode` of a probability density and create a Gaussian density distribution centered at this point. We will also need to find the curvature, or the precision, in the same point.

### Why do we need that?

Most of the time we are dealing with complicated probability distributions for which, for example, we do not have an easy way to sample or compute expectations. In cases like that may be worth it to use an approximation with an easy to handle probability distribution.

We can go beyond that. As we will see, La place Approximation can be used to transform a common neural network in a bayesian neural network. This will give us expectation and uncertainty.

### Developing Laplace Approximation

Suppose we have an unknown probability distribution over a one dimensional random variable $z$:

$$p(z) = \frac{f(z)}{Z}$$

where $Z = \int f(z)dz$. We want to find and Gaussian approximation $q(z)$ centered at the mode of $p(z)$. The mode is the point $z_{0}$ in which $\frac{df}{dz}\vert_{z=z_{0}}$, it is independent of the normalization factor $Z$.

Since we are making an approximation let us expand $f(z)$ or $\ln{f(z)}$ around $z_{0}$:

$$\ln{f(z)} = \sum_{n=0}^{\infty}\frac{(\ln{f(z)})^{(n)}}{n!}\vert_{z=z_{0}}(z-z_{0})^{n}$$

Expanding it until second order we have:

$$\ln{f(z)} \approx \left.\ln{f(z)}\right|_{z=z_{0}} + \left.\frac{1}{f(z)}\frac{df(z)}{dz}\right|_{z=z_{0}}(z-z_{0}) + \left.\left(\frac{-1}{f(z)^{2}} \left(\frac{df(z)}{dz} \right)^{2} + \frac{1}{f(z)} \frac{d^{2}f(z)}{dz^{2}} \right) \right|_{z=z_{0}} \frac{(z-z_{0})^{2}}{2!}$$

We know that $\frac{df}{dz}\vert_{z=z_{0}}$, which simplifies the equation

$$\ln{f(z)} \approx \ln{f(z_{0})} + \left.\frac{d^{2}}{dz}\ln{f(z)}\right|_{z=z_{0}} \frac{(z-z_{0})^{2}}{2!}$$

Now we exponentiate everything to get

$$f(z) \approx f(z_{0}) e^{\left(\frac{-A}{2} (z-z_{0})^{2} \right)}$$

where $A = \left.\frac{d^{2}}{dz}\ln{f(z)}\right|_{z=z_{0}}$. Note that we used 

$$\frac{d^{2}}{dz}\ln{f(z)} = \left(\frac{-1}{f(z)^{2}} \left(\frac{df(z)}{dz} \right)^{2} + \frac{1}{f(z)} \frac{d^{2}f(z)}{dz^{2}} \right).$$

Comparing the result with the Gaussian distribution one can see that $f(z_{0}) = \sqrt{\frac{A}{2 \pi}}$ and $A > 0$. We conclude that $A$ is the precision $A = \frac{1}{\sigma^{2}}$.

So $A$ is the curvature of a Gaussian at $z_{0}$. If the curvature goes to infinity, the standard deviation goes to zero, resulting in the Dirac distribution.

It will be more clear if we extend this to a multidimensional random variable, where $\mathbf{z}$ is the random variable vector and our constraint is $\nabla f(\mathbf{z})=0$ at point $\mathbf{z}_{0}$. The final equation is almost the same, but now we have a quadratic form:

$$\ln{f(z)} \approx \ln{f(z_{0})} - \frac{1}{2}(\mathbf{z}-\mathbf{z}_{0})^{\top}A (\mathbf{z}-\mathbf{z}_{0})$$

Since $f$ receives a multidimensional vector and returns a number (a scalar field), its gradient is a vector field and when we compute the second derivative (the Jacobian of the gradient) we are actually computing the Hessian matrix. In conclusion, $A$ is the Hessian matrix at point $\mathbf{z}_{0}$, the curvature matrix.

Finally, our approximation is

$$q(\mathbf{z}) = \frac{\left| A \right|^{\frac{1}{2}}}{\left(2 \pi \right)^{\frac{n}{2}}} exp \left( - \frac{1}{2}(\mathbf{z}-\mathbf{z}_{0})^{\top}A (\mathbf{z}-\mathbf{z}_{0})\right) = \mathcal{N}(\mathbf{z} | \mathbf{z}_{0}, A^{-1}),$$

where $\left| A \right|$ is the determinant of $A$.

### How to use it

We know an expression for $q(\mathbf{z})$. To apply the Laplace Approximation we just need to find de mode $\mathbf{z}_{0}$ with any numeric optimization algorithm and then use some algorithm tocompute the Hessian at this point.