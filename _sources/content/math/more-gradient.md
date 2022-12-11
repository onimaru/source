(more-gradient)=
# More on Gradients

Since there are many algorithms based on gradient descent, the gradient computation of many quantities can appear. So, here is some of common ones.

Let $w \in \mathbb{R}^{n}$ be a model parameter vector. Let $x \in \mathbb{R}^{n}$ a feature input vector.

1.  Let the map $L: \mathbb{R}^{n} \rightarrow \mathbb{R}$ be the loss function which maps parameter vectors to the real line, define as $L(w)=x^{\top}w=x_{i}w^{i}$. We write the gradient of $L$ as:

$$\begin{align}
  \nabla L(w) &= \frac{\partial}{\partial w_{j}} \left(x_{i}w^{i}\right) \\
              &= \partial^{j} x_{i}w^{i} \\
              &= \partial^{j}(x_{i})w^{i} + x_{i}\partial^{j}w^{i}\\
              &= 0 + x_{i} \delta^{jl}\partial_{l}w^{i} \\
              &= x_{i} \delta^{jl} \delta_{l}^{i} \\
              &= x_{i} \delta^{ji} \\
              &= x_{i} \delta^{ij} \\
              &= x^{j}
\end{align}$$

where the quantity $\delta_{l}^{i}=\frac{\partial w^{i}}{\partial w^{l}}$ is the Kronecker delta and $\delta^{jl}=\delta^{lj}$ is the Euclidian space metric tensor.

2. Let $X \in \mathbb{R}^{n \times n}$ be a square matrix of constant elements with respect to the model parameters. Let the loss function be defined as $L(w)= w^{\top}Xw$.

Let us rewrite the loss in index notation as $L(w) = w_{i}X^{i}_{j}w^{j}$, then its gradient is:

$$\begin{align}
  \nabla L(w) &= \frac{\partial}{\partial w_{k}} \left(w_{i}X^{i}_{j}w^{j}\right) \\
              &= \partial^{k} (w_{i}) X^{i}_{j} w^{j} + w_{i} X^{i}_{j} \partial^{k} w^{j} \\
              &= \delta^{k}_{i} X^{i}_{j} w^{j} + w_{i} X^{i}_{j} \delta^{kj} \\
              &= X^{k}_{j} w^{j} + w_{i} X^{ik} \\
              &= X^{k}_{j} w^{j} + \delta_{ij} X^{ik} w^{j} \\
              &= X^{k}_{j} w^{j} + X_{j}^{k} w^{j} \\
              &= \left(X^{k}_{j} + X_{j}^{k} \right) w^{j}
\end{align}$$

where $X_{j}^{k}$ is the transpose of $X^{k}_{j}$ and we also used $w_{i} = \delta_{ij}w^{j}$.

3. Let the loss function be defined as $L(w) = w^{\top}w$.

The function $L$ is an inner product, but since we treat the $w$ vectors as tensors, this is the same as an outer product: $w^{\top}w= w^{\top} \otimes w = w_{i} \delta^{i}_{j}w^{j}$ (I am sorry for the digression). Thus, the gradient is:

$$\begin{align}
  \nabla L(w) &= \frac{\partial}{\partial w_{k}} \left(w_{i}w^{i}\right) \\
              &= \left(\partial^{k} w_{i} \right) w^{i} + w_{i} \left(\partial^{k} w^{i} \right) \\
              &= \delta^{k}_{i} w^{i} + w_{i}\delta^{ki} \\
              &= w^{k} + w^{k} \\
              &= 2 w^{k}
\end{align}$$

4. Let the loss function be defined as the norm $L(w) = \vert \vert w \vert \vert_{2}$.

Here we need to use the chain rule of derivatives, since $\vert \vert w \vert \vert_{2} = \sqrt{w^{\top}w}$. So we have:

$$\begin{align}
  \nabla L(w) &= \frac{\partial}{\partial w_{k}} \left(w_{i}w^{i}\right)^{1/2} \\
              &= \frac{1}{2} \left(w_{i}w^{i}\right)^{-1/2} \partial^{k} \left(w_{i}w^{i}\right) \\
              &= \frac{1}{2 \vert \vert w \vert \vert_{2}} 2w^{k}\\
              &=  \frac{w^{k}}{\vert \vert w \vert \vert_{2}}
\end{align}$$

Each component of the gradient of the norm is the relation between the component and the norm, which makes sense.

5. Let the loss function be defined as the norm $L(w) = f\left(\vert \vert w \vert \vert_{2} \right)$, where $f$ is a function $f: \mathbb{R} \rightarrow \mathbb{R}$.

This is very similar to the last case, actually it is more general.

$$\begin{align}
  \nabla L(w) &= \frac{\partial}{\partial w_{k}} f\left(\vert \vert w \vert \vert_{2} \right) \\
              &= \frac{df}{d \vert \vert w \vert \vert_{2}} \partial^{k}\left(\vert \vert w \vert \vert_{2} \right) \\
              &= \frac{df}{d \vert \vert w \vert \vert_{2}} \frac{w^{k}}{\vert \vert w \vert \vert_{2}}
\end{align}$$

That is it for now. Sometimes when implementing some gradient based algorithm it is useful to know before hand if the gradient can be written explicitly. This can save us some time and not depend too much on auto-grad frameworks.
