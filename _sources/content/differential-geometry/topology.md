(Topology)=
# Topologia

1 - A distância de Minskowski, que é uma generalização de muitas métricas famosas, é também conhecida como distância $L^{p}$ e pode ser definida como:

$$d(x,y)= \left( \sum_{i=1}^{n} |x_{i}-y_{i}|^{p} \right)^{\frac{1}{p}},$$

onde $x,y \in \mathbb{R}^{n}$ e $p$ é um inteiro. Para $p$ igual a $1$, $2$ e $\infty$, essa distância se reduz à distâncias Manhattan ($L^{1}$), Euclidiana ($L^{2}$) e Chebyshev ($L^{\infty}$), respectivamente. Mostre que para $p=0.5$ a distância de Minkowski não é uma métrica (isso é válido para qualquer $p<1$).

```{dropdown} **Solução**: 
Sejam os pontos do $\mathbb{R}^{2}$: $x=(0,0)$, $y=(0,1)$ e $z=(1,1)$. Para $p=0.5$ temos

$$L^{\frac{1}{2}} \equiv d(x,y) = \left( \sum_{i=1}^{2} |x_{i}-y_{i}|^{\frac{1}{2}} \right)^{2}.$$

Verificando a desigualdade triangular, $d(x,z) \le d(x,y) + d(y,z)$, temos uma violação:

$$4 \le 1 + 1.$$

Logo, a distância $L^{\frac{1}{2}}$ não é uma métrica.
```
<br/>

2 - Seja $A$ qualquer conjunto. Mostre que a métrica discreta

$$ d(x,y) = \left\{\begin{matrix}
 0,& \text{se } x = y  \\
 1,& \text{se } x \ne y \\
\end{matrix}\right.$$

é uma métrica em $A$.

```{dropdown} **Solução**: 
Sejam $x,y,z \in A$. Como $x=y$ implica em $y=x$, a simetria é verdadeira, $d(x,y)=d(y,x)$. Como $d$ só pode assumir os valores $0$ ou $1$, $d(x,y) \ge 0$. Para a desigualdade triangular temos a seguintes possibilidades:

$$\begin{matrix}
 x \ne y \ne z \implies 1 \le 1+1;\\
 x \ne y = z \implies 1 \le 1+0;\\
 x = z \ne z \implies 0 \le 1+1\\
 x = y = z \implies 0 \le 0+0.
\end{matrix}$$

Logo, a métrica discreta é uma métrica.
```
<br/>

3 - A distância de Hausdorff é uma medida que quantifica quão longe dois subconjuntos de um espaço métrico estão um do outro. Mais formalmente, dada a métrica $ d $ em um espaço métrico $(X, d)$ e dois subconjuntos $A$ e $B$ de $X$, a distância de Hausdorff $d_{H}(A, B)$ é definida como:

$$
d_{H}(A, B) = \max \left\{ \sup_{a \in A} \inf_{b \in B} d(a, b), \sup_{b \in B} \inf_{a \in A} d(b, a) \right\},
$$

onde 

$$\sup_{a \in A} \inf_{b \in B} d(a, b)$$

significa que para cada ponto $ a $ em $A$, calculamos a menor distância até qualquer ponto em $B$ (isto é, a distância de $a$ ao ponto de $B$ mais próximo). Em seguida, tomamos o maior valor dessas menores distâncias para todos os pontos de $A$.

A distância de Hausdorff $d_{H}(A, B)$ é, portanto, o máximo entre essas duas quantidades. Ela representa a maior "distância mínima" que você deve percorrer para cobrir completamente um dos conjuntos a partir do outro. Se a distância de Hausdorff entre $A$ e $B$ é pequena, isso significa que os conjuntos $A$ e $B$ estão próximos um do outro em termos de sua posição no espaço métrico.

<br/>

4 - A distância de Wasserstein, também conhecida como Métrica de Transporte Ótimo ou Métrica de Earth Mover, é uma medida da diferença entre duas distribuições de probabilidade. Seja $\mu$ e $\nu$ duas distribuições de probabilidade definidas sobre um espaço métrico $(X, d)$. A distância de Wasserstein de ordem $p$, denotada por $W_p(\mu, \nu)$, é definida como:

$$
W_p(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{X \times X} d(x, y)^p \, d\gamma(x, y) \right)^{1/p}
$$

Aqui:

1. $d(x, y)$ é a distância entre os pontos $x$ e $y$ no espaço métrico $X$.

2. $\Gamma(\mu, \nu)$ é o conjunto de todas as medidas de acoplamento $\gamma$ entre $\mu$ e $\nu$, ou seja, o conjunto de distribuições conjuntas $\gamma(x, y)$ sobre $X \times X$ cujas marginais são $\mu$ e $\nu$. Isso significa que:

$$
\int_X \gamma(x, y) \, dy = \mu(x) \quad \text{e} \quad \int_X \gamma(x, y) \, dx = \nu(y)
$$

3. $p \geq 1$ é um parâmetro que determina a ordem da métrica de Wasserstein.

A interpretação intuitiva da distância de Wasserstein é a seguinte: ela representa o "custo mínimo" necessário para transformar uma distribuição $\mu$ na distribuição $\nu$, onde o "custo" de mover a "massa" de $x$ para $y$ é proporcional a $d(x, y)^p$.

Um caso especial importante é a distância de Wasserstein de ordem 1, $W_1(\mu, \nu)$, que é frequentemente utilizada em várias aplicações práticas.