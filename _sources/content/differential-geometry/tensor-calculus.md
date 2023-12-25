(tensor-calculus)=
# Cálculo Tensoarial

1 - Usando os axiomas de espaços vetoriais, mostre que é possível definir um espaço vetorial $V(2,\mathbb{R})$. Deixe claro quais são os elementos nulo, unitário e inverso.

```{dropdown} **Solução**:

 Definimos os vetores, $\textbf{u},\textbf{v},\textbf{w},\textbf{0} \in V(2,\mathbb{R})$:

$$\begin{align*}
\textbf{u} &= u^{1}e_{1}+u^{2}e_{2} = (u^{1}, u^{2})\\
\textbf{v} &= v^{1}e_{1}+v^{2}e_{2} = (v^{1}, v^{2})\\
\textbf{w} &= w^{1}e_{1}+w^{2}e_{2} = (w^{1}, w^{2})\\
\end{align*}$$

onde $e_{1}$ e $e_{2}$ são as bases de $V(2,\mathbb{R})$ e $u^{1},u^{2}v^{1},v^{2}w^{1},w^{2},0,1$ são escalares pertencentes ao corpo dos reais. Agora mostramos que esses vetores o obedecem os axiomas de formação de um espaço vetorial.

(i) - 

$$\begin{align*}
\textbf{u} + \textbf{v} &= \textbf{v} + \textbf{u}\\
(u^{1}+v^{1},u^{2}+v^{2}) &= (v^{1}+u^{1},v^{2}+u^{2})
\end{align*}$$

Como os escalares são reais, sua soma é comutativa.

(ii) - 

$$\begin{align*}
(\textbf{u} + \textbf{v}) + \textbf{w} &= \textbf{u} + (\textbf{v} + \textbf{w})\\
(u^{1}+v^{1},u^{2}+v^{2}) + (w^{1},w^{2}) &= (u^{1},u^{2}) + (v^{1}+w^{1},v^{2}+w^{2})\\
(u^{1}+v^{1}+w^{1},u^{2}+v^{2}+w^{2}) &= (u^{1}+v^{1}+w^{1},u^{1}+v^{2}+w^{2})\\
\end{align*}$$

Novamente a comutatividade da soma dos reais garante a associatividade da soma vetorial.

(iii) - seja o elemento nulo: $\textbf{0} = 0e_{1}+0e_{2} = (0, 0)$

$$\begin{align*}
\textbf{0} + \textbf{u} &= (0,0) + (u^{1},u^{2})\\
 &= (0+u^{1},0+u^{2}) \\
 &= (u^{1},u^{2})\\
 &= \textbf{u}
\end{align*}$$

(iv) - 

$$\begin{align*}
\textbf{u} + (-\textbf{u}) &= (u^{1},u^{2}) + (-u^{1},-u^{2})\\
 &= (u^{1}-u^{1},u^{2}-u^{2})\\
 &= (0,0)\\
 &= \textbf{0}
\end{align*}$$

(v) - seja $a \in \mathbb{R}$,

$$\begin{align*}
a(\textbf{u}+\textbf{v}) &= a\textbf{u}+ a\textbf{v}\\
a(u^{1}+v^{1},u^{2}+v^{2}) &=(au^{1},au^{2}) + (av^{1},av^{2})\\
(a(u^{1}+v^{1}),a(u^{2}+v^{2})) &= (au^{1}+av^{1},au^{2}+av^{2})\\
(au^{1}+av^{1},au^{2}+av^{2}) &= (au^{1}+av^{1},au^{2}+av^{2})
\end{align*}$$

(vi) - sejam $a,b \in \mathbb{R}$,

$$\begin{align*}
(a+b)\textbf{u} &= a\textbf{u}+ b\textbf{u}\\
 &= a(u^{1},u^{2}) + b(u^{1},u^{2})\\
 &=(au^{1},au^{2}) + (bu^{1},bu^{2})\\
 &= (au^{1}+bu^{1},au^{2}+bu^{2})\\
 &= ((a+b)u^{1},(a+b)u^{2})\\
 &= (a+b)\textbf{u}
\end{align*}$$

(vii) - sejam $a,b \in \mathbb{R}$,

$$\begin{align*}
(ab)\textbf{u} &= a(b\textbf{u})\\
 &= a(bu^{1},bu^{2})\\
 &=(abu^{1},abu^{2})\\
 &= (ab)(u^{1},u^{2})\\
 &= (ab)\textbf{u}
\end{align*}$$

Como o produto $ab$ é apenas um novo escalar $c \in \mathbb{R}$, não importa a ordem de atuação dos escalares.

(viii) - seja $1 \in \mathbb{R}$ o elemento unitário de $\mathbb{R}$,

$$\begin{align*}
1\textbf{u} &= 1(u^{1},u^{2})\\
 &= (1u^{1},1u^{2}) \\
 &= (u^{1},u^{2})\\
 &= \textbf{u}
\end{align*}$$
```

<br/>