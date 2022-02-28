(ranknet)=
# Ranknet

The algorithm called Ranknet is a way to train a neural network to assign a score to an observation in almost the same way we did manually in the previous section.

Let $\mathcal{D} = \{(\mathbf{x}_{i},y_{i}) | i \in \{1,2,...,N\}, \mathbf{x}_{i} \in \mathbf{X} \subset \mathbb{R}^{m} \text{ and } y_{i} \in \mathbb{R}^{1} \}$ be a dataset with $m$ features, 1 label and $N$ obervations.

Assuming we are solving a ranking problem, the label represents some kind of relevance. In this example we will assume that a label can have two values, $\{0,1\}$. So the definition above is changed to $y_{i} \in \{0,1\}$, where $1$ means more relevance than $0$. In future examples we will see other kinds of relevance. 

To train a Ranknet one needs in each iteration to sample a pair of observations, **one more relevant than the other**. We will refer to them as:

- $\mathbf{x}_{i,r}$: features vector of observation $i$ which has relevance $r$.

Since we know the relevance of an observation in sampling stage, we will refer to a relevant observation as $\mathbf{x}_{i,1}$ and $\mathbf{x}_{i,0}$ to the non-relevant one.

If you do not like the indices you can think the relevance being the label, such that $\mathbf{x}_{i,r}$ becomes the pair $\left( \mathbf{x}_{i},y_{i} \right)$. It does not matter the way we identify the relevance of an observation, just remember, again, to sample two of them **one more relevant than the other**.

Basically our model will be a map $f: \mathbf{X} \to \mathbb{R}^{1}$, i.e.,

$$f(\mathbf{x}_{i,r};\theta) = s_{i},$$

where $\theta$ are the set of parameters we want the model to learn. After that we use the same procedure as before: compute likelihood for each pairs, use the log-likelihood as an error and use the $\lambda{ij}$ to update the model.

## Ranknet algorithm

The algorithm to train our neural network is the following:

```{prf:algorithm} Ranknet algorithm
:label: ranknet-

**Init** weights $\theta$ for model $f$

Repeat for T epochs:  
- Sample $\mathbf{x}_{i,1},\mathbf{x}_{j,0} \sim \mathcal{D}$  
    - Train and update $f$:
        - Compute scores  
            - $s_{i,1} = f(\mathbf{x}_{i,1};\theta)$  
            - $s_{j,0} = f(\mathbf{x}_{j,0};\theta)$  
        - Compute likelihood, $\sigma(s_{i,1},s_{j,0})$  
            - $prob = sigmoid(s_{i,1},s_{j,0})$  
        - Compute loss:
            - $\mathcal{L} \left( prob,1 \right)$
        - Update $f$:
            - $\theta \leftarrow \theta + \gamma \lambda_{ij}$ 
```

The parameter $\gamma$ is the learning rate used to adjust the updating.

## Building and training the ranknet

To train out model we will use the {ref}`ranking teams dateset<ranking-dataset-teams>` defined in the datasets section.

```python
# Imports
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
```

```python
# loader function
class CustomDataset(Dataset):
    def __init__(self,x):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx]
    def __len__(self):
        return self.length

def create_loader(x,workers,batch_size):
    dataset = CustomDataset(x)
    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=workers)
    return loader

# neural network architecture
class Net(nn.Module):
    def __init__(self,input_dim=13,hidden_dim=64,output_dim=1,ngpu=ngpu):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden1 = nn.Linear(self.input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.hidden3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.out = nn.Linear(hidden_dim//4, self.output_dim)

    def forward(self, x):
        h = nn.Dropout(p=0.0)(torch.relu(self.hidden1(x)))
        h = nn.Dropout(p=0.0)(torch.relu(self.hidden2(h)))
        h = nn.Dropout(p=0.0)(torch.relu(self.hidden3(h)))
        return self.out(h)

# initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight).to(device)
        m.bias.data.fill_(0.001)
    
# function to define model
def define_model(input_dim=3,hidden_dim=64,output_dim=1,ngpu=ngpu):
    model = Net(input_dim,hidden_dim,output_dim).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        encoder = nn.DataParallel(encoder, list(range(ngpu)))
        decoder = nn.DataParallel(decoder, list(range(ngpu)))

    model.apply(init_weights)

    learning_rate, beta1, beta2 = [1e-4,0.9,0.999]
    eps,weight_decay,amsgrad = [1e-8,1e-3,False]
    optimizer = optim.Adam(params=model.parameters(),
                               lr=learning_rate,
                               betas=(beta1,beta2),
                               eps=eps,
                               weight_decay=weight_decay,
                               amsgrad=amsgrad)
    return model,optimizer

# function to split observations of X in (X_relevant,X_irrelevant)
def build_data_batch(X):
    rel_data = X[:,:input_dim].to(device)
    irel_data = X[:,input_dim:].to(device)
    return rel_data,irel_data

def ones_tensor(data_size):
    return torch.ones((data_size,1)).to(device)

# training function
def train_model(model,optimizer,loader,scheduler,epochs):
    loss_his = []
    loss_func = nn.BCELoss()
    for epoch in range(1,epochs+1):
        loss_ = []
        for i, X_train in enumerate(loader):
            rel_train,irel_train = build_data_batch(X_train)
            ones = ones_tensor(rel_train.shape[0])
            probs = torch.sigmoid(model(rel_train)-model(irel_train))
            loss = loss_func(probs,ones)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_his.append(loss.item())
        scheduler.step()
        if epoch%10 == 0:
            print(f"{epoch}| Train loss: {loss.item():.6f}")
    return model,loss_his
```

```python
team_df = pd.read_csv("data/ranking-teams-simulated.csv")
```

```python
workers      = 12
batch_size   = 13
input_dim,hidden_dim,output_dim = [7,100,1]
train_data = np.load("data/ranking-teams.npy")
train_loader = create_loader(train_data,workers,batch_size)

model,optimizer = define_model(input_dim,hidden_dim,output_dim,ngpu)
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=[lambda1],verbose=False)
```

```python
model,loss_his = train_model(model,optimizer,train_loader,scheduler,100)
```

```python
def compute_ranknet_scores(team_df,model):
    x = torch.tensor(team_df[["att","def","sta","coa","int","cre","luc"]].values,dtype=torch.float32).reshape(len(team_df),7).to(device)
    return model(x).detach().cpu().numpy()

team_df["score"] = compute_ranknet_scores(team_df,model)
team_df.sort_values(by="score",ascending=False)
```

```python
corr = team_df[["potencial","pontos","score"]].corr("spearman")
corr
```

```python
def dcg(r,i):
    return (2**r - 1)/(np.log2(1+i))

def dcg_k(x,k):
    result = 0
    for i in range(1,k+1):
        result += dcg(x[i-1],i)
    return result

def max_dcg_k(x,k):
    x = sorted(x)[::-1]
    return dcg_k(x,k)

def ndcg_k(x,k):
    return dcg_k(x,k)/max_dcg_k(x,k)

ndcg_list = []
for col in ["potential","points","score","att","def","sta","coa","int","cre","luc"]:
    ndcg_list.append([col]+[ndcg_k(team_df.sort_values(by=col,ascending=False)["potential"].values,k) for k in [3,10,20,26]])

ndcg_df = pd.DataFrame(ndcg_list,columns=["feat"]+[f"ndcg_{k}" for k in [3,10,20,26]])
```

```python
ndcg_df[ndcg_df["feat"].isin(["potential","points","score"])]
```

```python
ndcg_df[ndcg_df["feat"].isin(["att","def","sta","coa","int","cre","luc"])]
```