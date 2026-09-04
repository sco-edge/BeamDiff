"""Anchor-conditioned diffusion for BFA frame-to-frame circular deltas."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MOD=np.asarray([512,512,128,128],dtype=np.int32)

def signed_delta(x):
    raw=np.diff(x.astype(np.int32),axis=1)
    return ((raw+MOD//2)%MOD-MOD//2).astype(np.float32)

def reconstruct(anchor,delta):
    values=np.rint(delta).astype(np.int32)
    out=np.empty((len(anchor),10,234,4),dtype=np.uint16); out[:,0]=anchor
    for t in range(9):
        nxt=out[:,t].astype(np.int32)+values[:,t]
        nxt[...,:2]%=512
        nxt[...,2:]=np.clip(nxt[...,2:],0,127)
        out[:,t+1]=nxt
    return out

def anchor_features(a):
    phi=(a[...,:2].astype(np.float32)+.5)*(2*np.pi/512)
    psi=a[...,2:].astype(np.float32)/127*2-1
    return np.concatenate((np.cos(phi),np.sin(phi),psi),axis=-1)

def temb(t,dim):
    half=dim//2; f=torch.exp(torch.arange(half,device=t.device)*(-math.log(10000)/max(half-1,1)))
    z=t.float()[:,None]*f[None]; return torch.cat((z.sin(),z.cos()),1)

class Block(nn.Module):
    def __init__(self,w,c):
        super().__init__(); self.n1=nn.GroupNorm(8,w);self.n2=nn.GroupNorm(8,w)
        self.c1=nn.Conv2d(w,w,3,padding=1);self.c2=nn.Conv2d(w,w,3,padding=1);self.fc=nn.Linear(c,w)
    def forward(self,x,c):
        h=self.c1(nn.functional.silu(self.n1(x)))+self.fc(c)[:,:,None,None]
        return x+self.c2(nn.functional.silu(self.n2(h)))

class Model(nn.Module):
    def __init__(self,w=64,c=128):
        super().__init__();self.c=c;self.label=nn.Embedding(20,c)
        self.time=nn.Sequential(nn.Linear(c,c),nn.SiLU(),nn.Linear(c,c))
        self.inp=nn.Conv2d(10,w,3,padding=1);self.blocks=nn.ModuleList([Block(w,c) for _ in range(6)])
        self.out=nn.Sequential(nn.GroupNorm(8,w),nn.SiLU(),nn.Conv2d(w,4,3,padding=1))
    def forward(self,x,t,y,a):
        # Repeat the first-frame anchor along the nine delta positions.
        a=a.permute(0,3,1,2).repeat(1,1,9,1)
        h=self.inp(torch.cat((x,a),1)); c=self.time(temb(t,self.c))+self.label(y)
        for b in self.blocks:h=b(h,c)
        return self.out(h)

def sched(n,device):
    b=torch.linspace(1e-4,.02,n,device=device);a=1-b;ab=torch.cumprod(a,0);return b,a,ab

@torch.no_grad()
def generate(model,y,anchor,n,device):
    b,a,ab=sched(n,device);x=torch.randn(len(y),4,9,234,device=device)
    for i in reversed(range(n)):
        t=torch.full((len(y),),i,dtype=torch.long,device=device);e=model(x,t,y,anchor)
        x=(x-(1-a[i])/torch.sqrt(1-ab[i])*e)/torch.sqrt(a[i])
        if i:x+=torch.sqrt(b[i])*torch.randn_like(x)
    return x.cpu().numpy()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output_dir',type=Path)
    p.add_argument('--split-indices-dir',type=Path);p.add_argument('--split-seed',type=int,default=111)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=20);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--max-samples',type=int)
    p.add_argument('--generated-per-class',type=int,help='generate this many samples for every activity; scarce anchors are reused with new diffusion noise')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    ck=args.output_dir/'checkpoint_latest.pt'
    if args.output_dir.exists() and not args.resume:p.error('output exists; use --resume or another output')
    if args.resume and not ck.is_file():p.error(f'missing {ck}')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(args.input,allow_pickle=False) as f:
        x=f['x'];y=f['y'].astype(np.int64);meta={k:f[k] for k in ('source','window_start','participant')}
    train=(np.load(args.split_indices_dir/'train_indices.npy').astype(np.int64) if args.split_indices_dir
           else np.flatnonzero(np.random.default_rng(args.split_seed).random(len(y))<.70))
    if args.max_samples:
        rng=np.random.default_rng(args.seed);z=[];per=max(1,args.max_samples//20)
        for label in range(20):
            c=train[y[train]==label];z.extend(rng.choice(c,min(per,len(c)),replace=False))
        train=np.asarray(z,dtype=np.int64)
    raw=signed_delta(x[train]);mean=raw.mean((0,1,2));std=np.maximum(raw.std((0,1,2)),1e-3)
    normalized=(raw-mean)/std
    anchors=anchor_features(x[train,:1])
    ds=TensorDataset(torch.from_numpy(normalized).permute(0,3,1,2),torch.from_numpy(y[train]),torch.from_numpy(anchors))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    model=Model().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate);_,_,ab=sched(args.steps,device)
    args.output_dir.mkdir(parents=True,exist_ok=True);history=[];start=0
    if args.resume:
        s=torch.load(ck,map_location=device,weights_only=False);model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
        history=s['history'];start=s['epoch']
        if not np.array_equal(s['train_indices'],train):p.error('train indices changed')
        mean,std=s['mean'],s['std'];print(f'Resuming after epoch {start}',flush=True)
    for epoch in range(start,args.epochs):
        losses=[];model.train()
        for clean,label,anchor in loader:
            clean,label,anchor=clean.to(device),label.to(device),anchor.to(device)
            t=torch.randint(args.steps,(len(clean),),device=device);noise=torch.randn_like(clean);q=ab[t][:,None,None,None]
            noisy=torch.sqrt(q)*clean+torch.sqrt(1-q)*noise;loss=nn.functional.mse_loss(model(noisy,t,label,anchor),noise)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        history.append(float(np.mean(losses)));torch.save(dict(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),
            history=history,train_indices=train,mean=mean,std=std),ck);print({'epoch':epoch+1,'loss':history[-1]},flush=True)
    if args.generated_per_class:
        generation_rng=np.random.default_rng(args.seed+1);generation=[]
        for label in range(20):
            candidates=train[y[train]==label]
            if not len(candidates):p.error(f'no training anchor available for class {label}')
            generation.extend(generation_rng.choice(candidates,args.generated_per_class,
                replace=len(candidates)<args.generated_per_class))
        generation=np.asarray(generation,dtype=np.int64)
    else:generation=train.copy()
    generation_labels=y[generation];generation_anchors=anchor_features(x[generation,:1])
    generation_suffix=(f'_per_class_{args.generated_per_class}' if args.generated_per_class else '')
    chunks=args.output_dir/f'generation_chunks{generation_suffix}';chunks.mkdir(exist_ok=True);made=[]
    for begin in range(0,len(generation),args.batch_size):
        end=min(begin+args.batch_size,len(generation));path=chunks/f'{begin:08d}_{end:08d}.npy'
        if path.is_file():out=np.load(path,allow_pickle=False)
        else:
            label=torch.from_numpy(generation_labels[begin:end]).to(device);anchor=torch.from_numpy(generation_anchors[begin:end]).to(device)
            delta=generate(model,label,anchor,args.steps,device).transpose(0,2,3,1)*std+mean
            out=reconstruct(x[generation[begin:end],0],delta);tmp=path.with_suffix('.tmp.npy');np.save(tmp,out);tmp.replace(path)
            print({'generated':end,'total':len(generation)},flush=True)
        made.append(out)
    generated_path=args.output_dir/f'generated_bfa{generation_suffix}.npz'
    protocol_path=args.output_dir/f'protocol{generation_suffix}.json'
    made=np.concatenate(made);np.savez_compressed(generated_path,x=made,y=generation_labels,
        **{k:v[generation] for k,v in meta.items()},allocation_real_index=generation,augmentation_eligible=np.ones(len(generation),bool),
        train_split_seed=np.asarray(args.split_seed),augmentation=np.asarray('anchor-conditioned-bfa-delta-ddpm-v1'))
    protocol_path.write_text(json.dumps(dict(input=str(args.input),device=str(device),seed=args.seed,
        split_seed=args.split_seed,steps=args.steps,epochs=args.epochs,train_samples=len(train),generated_samples=len(generation),
        generated_per_class=args.generated_per_class,generated_class_counts=np.bincount(generation_labels,minlength=20).tolist(),
        unique_generation_anchors=len(np.unique(generation)),delta_mean=mean.tolist(),
        delta_std=std.tolist(),loss=history),indent=2));print('Saved',made.shape,flush=True)

if __name__=='__main__':main()
