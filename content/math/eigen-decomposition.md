(matrix-decomposition)=
# Decomposição de matrizes

## Eigen Decomposition

Sejam um vetor $v$ e uma matriz $\mathbf{A}$. $v$ é um autovetor de $\mathbf{A}$ se o produto $\mathbf{A}v$ alterar apenas o tamanho de $v$:

$$\mathbf{A}v = \lambda v$$

onde $\lambda$ é um escalar chamado de autovalor correspondendo ao vetor $v$.

A utilidade disso envolve representar a matriz de outra maneira que possa ser útil, essa é a `eigen decomposition`.

Se $\mathbf{A}$ tiver $n$ autovetores linearmente independentes com seus respectivos autovalores. Podemos recriar a matriz original com a operação:

$$\mathbf{A} = \mathbf{V} \mathbf{ \Lambda } \mathbf{V}^{-1},$$

onde $\mathbf{V}$ é a matriz criada concatenando todos os autovetores coluna e $\mathbf{\Lambda}$ é a matriz diagonal com cada um dos autovalores na diagonal principal.

```python
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1.16,0.58],[0.25,1.08]])

# Compute the eigenvalues and right eigenvectors of a square array
λ,V =  LA.eig(A)
```

```python
np.diag(λ)
>>> array([[1.50288379, 0.        ],
           [0.        , 0.73711621]])
```

```python
v
>>> array([[ 0.86082477, -0.80802933],
           [ 0.50890149,  0.58914226]])
```

```python
# reconstruindo A
V @ np.diag(λ) @ LA.inv(V)
>>> array([[1.16, 0.58],
           [0.25, 1.08]])
```

```python
A @ V.T[0]
>>> array([1.29371959, 0.7648198 ])
```

Na figura a seguir vemos que o vetor $v_{1}$ é um autovetor de $\mathbf{A}$ pois aponta na mesma direção de $\mathbf{A}v_{1}$, ou seja, é uma transformação linear que não rotaciona o vetor, só aumenta seu tamanho. O mesmo não acontece com o vetor $w$, então este não é um autovetor de $\mathbf{A}$.

```python
v1 = A @ V.T[0]
w1 = np.array([0.5,0.8])
Aw1 = A @ w1

plt.arrow(x=0,y=0,dx=V.T[0][0],dy=V.T[0][1],length_includes_head=True,width=0.01,shape="full",label=r"$v$",color="blueviolet",alpha=0.5)
plt.arrow(x=0,y=0,dx=v1[0],dy=v1[1],length_includes_head=True,width=0.01,shape="full",label=r"$Av$",color="deepskyblue",alpha=0.5)
plt.arrow(x=0,y=0,dx=w1[0],dy=w1[1],length_includes_head=True,width=0.01,shape="full",label=r"$w$",color="deeppink",alpha=0.5)
plt.arrow(x=0,y=0,dx=Aw1[0],dy=Aw1[1],length_includes_head=True,width=0.01,shape="full",label=r"$Aw$",color="steelblue",alpha=0.5)
plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.tight_layout()
plt.show()
```