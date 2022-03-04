(reinforcement-learning)=
# Beyond conventional machine learning

🏗️


## Processos de Decisao de Markov Finitos (finite MDPs)

MDPs são a formalização clássica da tomada de decisão sequential, onde ações influenciam não só as recompensas imediatas, mas também  situações subsequentes, ou estados, e recompensas futuras.

## A interface Agente-Ambiente

Aquele que aprende e é o tomador de decisões é chamado de agente (agent). a coisa com a qual ele interage, composta de tudo aquilo fora do agente, é chamado de ambiente (environment).

A figura abaixo mostra a forma da interação entre os dois.

**Inserir imagem**

Os dois interagem a cada sequência de passos de um tempo discreto, $t$. A cada passo $t$ o agente recebe uma representação do estado do ambiente, $S_{t}\in \mathcal{S}$, e baseado nisso seleciona uma ação $A_{t}\in \mathcal{A}(s)$. No passo de tempo seguinte o agente, como consequência de sua ação, recebe um valor numérico como recompensa, $R_{t+1}\in \mathcal{R} \subset \mathbb{R}$, e se encontra em um novo estado, $S_{t+1}$. O MDP e o agente juntos dão assim origem a uma sequência ou trajetória como esta:

$$S_{0},A_{0},R_{1},S_{1},A_{1},R_{2},S_{2},A_{2},R_{3},...$$

Em um MDP finito os conjuntos $\mathcal{S},\mathcal{A}$ e $\mathcal{R}$ têm número de elementos finitos. Nesse caso, as variáveis aleatórias $R_{t}$ e $S_{t}$ têm definidas distribuições de probabilidade discretas dependente somente do estado e da ação anteriores. Assim:

$$p(s^{'},r \vert s,a) = P \lbrace S_{t}=s^{'},R_{t}=r \vert S_{t-1}=s,A_{t-1}=a \rbrace,\ \forall s^{'},s \in \mathcal{S}, r \in \mathcal{R}\ \text{e } a \in \mathcal{A}.$$

Perceba que como $p$ é uma probabilidade:

$$\sum_{s^{'}} \sum_{r} p(s^{'},r \vert s,a) = 1,\ \forall s \in \mathcal{S} \text{ e } a \in \mathcal{A}.$$

Em um MDP a probabilidades dadas por $p$ caracterizam completamente a dinâmica do ambiente e através dela podemos obter outras quantidades como:
- Probabilidades de transição de estado, $p:\mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$

$$p(s^{'} \vert s,a) = P \lbrace S_{t}=s^{'}\vert S_{t-1}=s,A_{t-1}=a \rbrace = \sum_{r}p(s^{'},r \vert s,a)$$

- Recompensa esperada para o par estado-ação, $r:\mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$

$$r(s,a) = \mathbb{E}\left[ R_{t} \vert S_{t-1}=s,A_{t-1}=a \right] = \sum_{r} r \sum_{s^{'}} p(s^{'},r \vert s,a)$$

- Recompensa esperada para a tripla estado-ação-estado seguinte, $r:\mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$

$$r(s,a,s^{'}) = \mathbb{E}\left[ R_{t} \vert S_{t-1}=s,A_{t-1}=a, S_{t}=s^{'} \right] = \sum_{r} r \frac{p(s^{'},r \vert s,a)}{p(s^{'} \vert s,a)}.$$

Em resumo o MDP é um framework abstrato e flexível que pode ser aplicado a problemas variados de diversas maneiras.

## Retornos e episódios


Em geral se quer que o agente maximize uma quantidade chamada retorno esperado (expectede return), $G_{t}$, que é definida como alguma função da sequência de recompensas que pode ser apenas uma soma
$$G_{t} = R_{t+1}+R_{t+2}+R_{t+3}+...+R_{T},$$
onde $T$ é o passo final.

No caso de uma tarefa não episódica, tarefa contínua, precisamos de um conceito adicional, o retorno descontado:
$$G_{t} = R_{t+1}+\gamma R_{t+2}+ \gamma^{2} R_{t+3}+... = \sum_{k=0}^{\infty} \gamma^{k}R_{t+k+1},$$
onde $\gamma$ é um parâmetro, $0 \le \gamma \le 1$, chamado taxa de desconto.

Podemos obter uma relação útil com
$$G_{t} = R_{t+1}+\gamma R_{t+2}+ \gamma^{2} R_{t+3}+...$$
$$G_{t} = R_{t+1}+\gamma \left( R_{t+2}+ \gamma R_{t+3}+... \right)$$
$$G_{t} = R_{t+1}+\gamma G_{t+1}$$
Assim o retorno descontado pode também ser usado em tarefas episódicas. Indo mais além podem definir:
$$G_{t} = \sum_{k=t+1}^{T}\gamma^{k-t-1} R_{k}$$
Assim $T$ pode ser infinito ou $\gamma =1$, mas ão ambos.

## Políticas e funções de valor


Em vários algoritmos precisamos de funções para estimar o valor de um estado ou do par estado-ação. Seu papel é fornecer informação sobre o quanto a entrada é valiosa para o agente. Essa funções de valor são definidas de modo a respeitar algum padrão de comportamento, chamado de política (policy).

Formalmente uma política é um mapeamento de estados para probabilidades de selecionar cada possível ação. Se um agente segue uma política $\pi$ no tempo $t$, então $\pi(a \vert s)$ é a probabilidade de $A_{t}=a$ se $S_{t}=s$.

Denotamos a função de valor de um estado sob a política $\pi$ como $v_{\pi}(s)$. Essa é o retorno esperado se inicia-se em $s$ seguindo a política $\pi$ após isso:
$$v_{\pi}(s) = \mathbb{E}_{\pi} \left[ G_{t} \vert S_{t}=s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \vert S_{t} \right],\ \forall s \in \mathcal{S}.$$
Desta maneira o valor do estado final é sempre zero. Utilizando umas das relações acima podemos escrever:
$$v_{\pi}(s) = \sum_{a} \pi(a \vert s) \sum_{s^{'},r} p(s^{'},r \vert s,a)\left[r + \gamma v_{\pi}(s^{'}) \right]$$
que é chamada de equação de Bellman para $v_{\pi}$.

De modo similar, o valor de tomar a ação $a$ em um estado $s$, seguindo a política $\pi$, $q_{\pi}(s,a)$, é:

$$q_{\pi}(s,a) = \mathbb{E}_{\pi} \left[ G_{t} \vert S_{t}=s,A_{t}=a \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \vert S_{t}, A_{t}=a \right],\ \forall s \in \mathcal{S}.$$

Usualmente $v_{\pi}(s)$ é chamada de função de valor de estado e $q_{\pi}(s,a)$ de função de valor de estado-ação. Essas quantidades podem ser estimadas através da experiência do agente.


---

Uma policy pode ser determinística, denotada por $\mu$:

$$a_{t}=\mu_{\theta}(s_{t})$$

ou estocástica, usualmente denotada por $\pi$:

$$a_{t} \sim \pi_{\theta}(\cdot \vert s_{t})$$

O símbolo $\theta$ representa os parâmetros das policies, usualmente são pesos de uma rede neural.

## Policies determinísticas

Exemplo: uma rede neural com duas camadas de tamanho 64 e ativação `tanh`. 
```python
pi_net = nn.Sequential(
              nn.Linear(obs_dim, 64),
              nn.Tanh(),
              nn.Linear(64, 64),
              nn.Tanh(),
              nn.Linear(64, act_dim)
            )
```

Se `obs` for uma matriz do numpy contendo um batch de observações, podemos obter as ações da seguinte maneira:

```python
obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi_net(obs_tensor)
```
## Policies estocásticas

Os tipos mais comuns são policies categoricas (para espaços de ação discretos) e gaussianas diagonais (para espaços de ação contínuos).

Dois cálculos são importantes para usar e treinar policies estocásticas:
- amostrar (sampling) ações da policy
- calcular log likelihoods de ações particulares, $\log{\pi_{\theta}(a \vert s)}$.

### Policies categóricas

*Sampling*: dadas as probabilidades para cada ação, frameworks como PyTorch e Tensorflow possuem ferramentas para fazer a amostragem.

*Log-Likelihood*: Denotando a última camada de probabilidades como $P_{\theta}(s)$. É um vetor com tantas entradas quantas forem as ações, então as ações podem ser tratadas como índices desse vetor. A log likelihood para uma ação `a` pode ser obtida indexando o vetor:

$$\log{\pi_{\theta}(a \vert s_{t})} = \log{P_{\theta}(s)}_{a}$$

### Policies gaussianas diagonais

Uma distribuição gaussiana multivariada é descrita por um vetor médio, $\mu$, e uma matriz de covariância, $\Sigma$. No caso da distribuição diagonal a matriz de covariância tem apenas entradas na diagonal principal.

Uma policy desse tipo tem uma rede neural que mapeia observações para ações médias, $\mu_{\theta}(s)$. A variância usualmente é transformada no log dos desvios padrões que, por sua vez, pode ser um parâmetro específico ou ser tb mapeada por uma rede neural.

*Sampling*: Dados o vetor médio, o desvio padrão e um vetor $z$ de ruído de uma esfera gaussiana, $z \sum \mathbb{N}(0,I)$, a amostragem de uma ação pode ser calculada com:

$$a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z$$

*Log-Likelihood*: A log-likelihood de uma ação $a$ $k$-dimensional pode ser calculada por:

$$\log{\pi_{\theta}(a \vert s_{t})} = \frac{-1}{2}\left(\sum_{i=1}^{k} \left(\frac{(a_{i}-\mu_{i})^{2}}{\sigma_{i}^{2}} +2 \log{\sigma_{i}} \right) +k \log{2 \pi}\right)$$

## Trajetórias

Uma trajetória, $\tau$, é uma sequência de estados e ações: 


$$\tau = (s_{0},a_{0},s_{1},a_{1},...)$$

As transições de estado podem ser determinísticas, 

$s_{t+1} = f(s_{t},a_{t})$

ou estocásticas

$s_{t+1} \sim P(\cdot \vert s_{t},a_{t})$

