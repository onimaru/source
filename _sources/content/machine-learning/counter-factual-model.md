(counter-factual-model)=
# Counter Factual Model
---

# Inferência Causal

É comum a confusão entre causa e associação e isso pode levar a muitas interpretações errôneas. Supomos que *X* e *Y* representam a distribuição de duas variáveis aleatórias, *x* e *y*. Ao afirmar "*X causa Y*" estamos dizendo que mudar o valor de *X* irá mudar o valor da distribuição *Y*. Se "*X causa Y*", *X* e *Y* estão associados, mas o contrário não é necessariamente verdade. Vamos analisar sobre maneiras de abordar causalidade: modelos contrafactuais e modelos gráficos.

## Modelo Contrafactual

Usaremos o dataset abaixo no qual temos algumas características de produtos de um e-Commerce juntamente com sua probabilidade de venda.
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/counter_factual_data_example.csv')
df['Sold'] = np.round(df['Selling_probability'],0)
df.head()
```
Neste dataset alguns produtos foram vendidos com frete grátis, `Free_Shipment = 1` e outros não `Free_Shipment = 0`. Estamos interessados em saber se oferecer frete grátis aumenta a probabilidade de venda de um produto, ou seja, `Free_Shipment` causa `Sold`. Os dados que possímos são dados observados e portanto não podemos voltar no tempo e oferecer frete grátis e observar novamente.

Usaremos `Free_Shipment` como a variável binária $X$ e `Sold` como a variável de saída $Y$. É comum chamar $X$ de variável de tratamento e essa abordagem de *efeito pós tratamento*. Então precisamos diferenciar as afirmações:

- "*X* causa *Y*";
- "*X* está associado a *Y*".

Para isso vamos utilizar duas variáveis aleatórias chamadas de **potenciais resultados**. São elas:

- $C_{0}$ - o resultado se $X=0$;
- $C_{1}$ - o resultado se $X=1$.

Assim, $Y = C_{0}$ se $X = 0$ e $Y = C_{1}$ se $X = 1$, ou ainda a chamada **relação de consistência**,

$Y = C_{X}$.

Vamos fazer isso no nosso dataset.
```python
df['C0'] = df[df['Free_Shipment']==0]['Sold']
df['C1'] = df[df['Free_Shipment']==1]['Sold']

df[['Free_Shipment','Sold','C0','C1']].head(10)
```
Os `NaN` indicam valores não observados. Se $X=0$ não temos como observar $C_{1}$, por isso diz-se que $C_{1}$ é **contrafactual** já que ele representa qual seria o resultado, contra os fatos, se $X=1$.

Estas novas variáveis podem ser vistas como variáveis ocultas encontradas em outros modelos.

Agora podemos definir o **efeito causal médio** ou **efeito de tratamento médio**. Este é dado por

$$\theta  = \mathbb{E}(C_{1})-\mathbb{E}(C_{0}).$$

Podemos interpretar $\theta$ como o valor esperado dos resultados caso todos os produtos tivessem $X=1$ menos o valor esperado dos resultados caso todos os produtos tivessem $X=0$. Há algumas maneiras de medir o efeito causal, por exemplo, se $C_{1}$ e $C_{0}$ são binários usualmente define-se a razão causal provável

$$\frac{\mathbb{P}(C_{1}=1)}{\mathbb{P}(C_{1}=0)} \div \frac{\mathbb{P}(C_{0}=1)}{\mathbb{P}(C_{0}=0)}$$

e o risco causal relativo

$$\frac{\mathbb{P}(C_{1}=1)}{\mathbb{P}(C_{0}=1)}$$

```python
# efeito causal médio
ecm = df['C1'].sum() / df.shape[0] - df['C0'].sum() / df.shape[0]

# razao causal provável
p_c1_1 = df['C1'].sum() / df.shape[0]
p_c1_0 = 1 - p_c1_1
p_c0_1 = df['C0'].sum() / df.shape[0]
p_c0_0 = 1 - p_c0_1

rcp = (p_c1_1/p_c1_0)/(p_c0_1/p_c0_0)

# risco causal relativo
rcr = p_c1_1 / p_c0_1

print("ECM: {:.3f}| RCP: {:.3f}| RCR: {:.3f}".format(ecm,rcp,rcr))
```
Já a associação é definida como

$$\alpha = \mathbb{E}(Y|X=1) - \mathbb{E}(Y|X=0),$$

ou seja, o valor esperado de $Y$ dado que $X=1$ menos o valor esperado de $Y$ dado que $X=0$. Vejamos a associação para nossos dados:
```python
association = df[df['Free_Shipment']==1]['Sold'].mean() - df[df['Free_Shipment']==0]['Sold'].mean()
print("Association: {:.3f}".format(association))
```
Verificamos uma associação positiva, ou seja, oferecer frete $X=1$ grátis aumenta as chances de venda, mas a associação não é igual ao efeito causal médio, como geralmente é o caso.

O que fazemos em machine learning usando aprendizado supervisionado está voltado à associação. Coletamos observações independentes umas das outras e encontramos uma função que generaliza as distribuições das features para encontrar $p(Y\|X_{1},...,X_{n})$. Para entender a causalidade precisamos encontrar algo diferente, $p(Y\|X_{1}=x,...,X_{n})$, a distribuição de $Y$ dado que $X_{1}$ tem valor igual a $x$ para todas as observações.
