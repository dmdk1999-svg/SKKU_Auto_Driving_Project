"""The reference architecture is state_dim -> 256 -> 256 -> 2 with input normalization."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
torch.set_num_threads(2)

class BC(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc1=nn.Linear(dim,256);self.fc2=nn.Linear(256,256);self.fc3=nn.Linear(256,2)
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('std',torch.ones(dim))
    def forward(self,x):
        x=(x-self.mean)/self.std
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    @torch.no_grad()
    def act(self,x):
        return np.clip(self(torch.as_tensor(x,dtype=torch.float32)).numpy(),-1,1)
    def save(self,path):
        torch.save({'dim':len(self.mean),'state':self.state_dict()},path)
    @classmethod
    def load(cls,path):
        ck=torch.load(path,weights_only=True,map_location='cpu')
        model=cls(ck['dim']);model.load_state_dict(ck['state']);model.eval()
        return model

def train(x,y,vx,vy,path,seed=0,epochs=50,initial=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=BC(x.shape[1]) if initial is None else BC.load(initial)
    if initial is None:
        model.mean.copy_(torch.tensor(x.mean(0)))
        sd=x.std(0);sd=np.where(sd<1e-6,1,sd)
        model.std.copy_(torch.tensor(sd))
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    x,y,vx,vy=[torch.as_tensor(a,dtype=torch.float32) for a in (x,y,vx,vy)]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[]
    for epoch in range(epochs):
        model.train();total=0
        for ids in torch.randperm(len(x)).split(256):
            loss=nn.functional.mse_loss(model(x[ids]),y[ids])
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=loss.item()*len(ids)
        model.eval()
        with torch.no_grad():
            loss=sum(nn.functional.mse_loss(model(a),b,reduction='sum').item()
                     for a,b in zip(vx.split(4096),vy.split(4096)))/vy.numel()
        history.append({'epoch':epoch+1,'train_mse':total/len(x),'validation_mse':loss})
        if loss<best:
            best=loss;model.save(path)
        if (epoch+1)%10==0: print(path.stem,seed,epoch+1,total/len(x),loss,flush=True)
    path.with_suffix('.training.json').write_text(json.dumps(history,indent=2))
    return BC.load(path)

