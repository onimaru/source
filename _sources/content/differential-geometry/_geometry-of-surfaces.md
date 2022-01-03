# Geometria das superfícies


A geometria de um recorte de um material, uma superfície, pode ser descrita pelo mapa $f: M \rightarrow \mathbb{R}^{3}$ de uma região $M$ no plano Euclidiano, $\mathbb{R}^{2}$ para um subconjunto $f(M)$ no $\mathbb{R}^{3}$.

O diferencial de tal mapa, denotado por $df$ nos diz como mapear um vetor $X$ no plano para o correspondente vetor $df(X)$ na superfície. O diferencial apenas nos diz como "push forward" vetores de um espaço para o outro.

O comprimento de um vetor tangente $X$ pode ser encontrado com $\sqrt{\langle X,X \rangle}$ no plano, enquanto na superfície fazemos $\sqrt{\langle df(X), df(X) \rangle}$. Já para encontrar o produto interno entre dois vetores tangentes $X$ e $Y$ fazemos: 

$$g(X,Y) = df(X) \cdot df(Y),$$

onde o mapa $g$ é a métrica da superfície induzida por $f$, ou apenas métrica.

Se $X$ for um vetor tangente e $u \in \mathbb{R}^{3}$ um vetor normal à superfície em um ponto $p$, temos:

$$df(X) \cdot u = 0$$

Isso é válido para qualquer vetor tangente em $p$. Denotamos o vetor normal unitário como $N$. Para superfícies orientáveis N é um mapa contínuo $N: M \rightarrow S^{2}$, chamado mapa de Gauss, que associa cada ponto da superfície com seu normal unitário, visto como um ponto da esfera $S^{2}$. O diferencial $dN$, chamado mapa de Weingarten, diz como o normal unitário muda de um ponto a outro. Para saber como $N$ muda ao longo de uma direção tangente $X$, avaliamos $dN(X)$.

### Coordenada conformes

Algumas vezes queremos permitir que o tamanho dos vetores permaneça constante ao serem "push-forward". Essa é a ideia de parametrização isométrica, que pode ser por comprimento de arco ou por unidade de velocidade. Para fazer isso o requisito é $\vert df(X) \vert  = \vert X \vert$, ou seja, preservamos a norma de qualquer vetor.

Dizemos que um mapa é conforme se preserva o ângulo entre quaisquer dois vetores. Assim, o mapa $f: \mathbb{R}^{2} \supset M \rightarrow \mathbb{R}^{3}$ satisfaz

$$df(X) \cdot df(X) = a \langle X,Y \rangle$$

para todos os vetores tangentes $X$ e $Y$ e onde $a$ é uma função positiva. É comum usar $a$ como $e^{u}$ para uma função real $u$. Isso não garante que o comprimento de vetores sejam preservados, mas garante que vetores ortogonais se mantenham ortogonais ou garante o paralelismo entre linhas. Mapas conformes sempre existem, entretanto os mapas isométricos podem não existir.