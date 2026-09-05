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

def continuous_reconstruct(anchor,delta):
    """Differentiably reconstruct BFA frames from a raw first frame and deltas."""
    frames=[anchor]
    current=anchor
    for step in range(delta.shape[1]):
        value=current+delta[:,step]
        phi=torch.remainder(value[...,:2],512.0)
        psi=value[...,2:].clamp(0.0,127.0)
        current=torch.cat((phi,psi),dim=-1)
        frames.append(current)
    return torch.stack(frames,dim=1)

def temporal_fidelity_loss(predicted,clean,std_weight=1.0,tail_weight=0.5):
    """Match typical, variable, and high-motion normalized BFA deltas."""
    predicted_abs=predicted.abs();clean_abs=clean.abs()
    sample_mean=nn.functional.smooth_l1_loss(
        predicted_abs.mean((2,3)),clean_abs.mean((2,3))
    )
    predicted_channel=predicted_abs.permute(1,0,2,3).reshape(4,-1)
    clean_channel=clean_abs.permute(1,0,2,3).reshape(4,-1)
    spread=nn.functional.smooth_l1_loss(
        predicted_channel.std(1,unbiased=False),clean_channel.std(1,unbiased=False)
    )
    tail=nn.functional.smooth_l1_loss(
        torch.quantile(predicted_channel,.95,dim=1),
        torch.quantile(clean_channel,.95,dim=1),
    )
    return sample_mean+std_weight*spread+tail_weight*tail,sample_mean,spread,tail

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

class BeamSenseTeacher(nn.Module):
    """PyTorch equivalent of the public BeamSense Keras CNN for input gradients."""
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(4,128,3,padding=1)
        self.conv2=nn.Conv2d(128,128,3,padding=1)
        self.bn1=nn.BatchNorm2d(128,eps=1e-3)
        self.conv3=nn.Conv2d(128,64,3,padding=1)
        self.conv4=nn.Conv2d(64,64,3,padding=1)
        self.bn2=nn.BatchNorm2d(64,eps=1e-3)
        self.conv5=nn.Conv2d(64,32,3,padding=1)
        self.conv6=nn.Conv2d(32,32,3,padding=1)
        self.bn3=nn.BatchNorm2d(32,eps=1e-3)
        self.fc=nn.Linear(2*234*32,20)

    def forward(self,x):
        x=x.permute(0,3,1,2)
        x=nn.functional.relu(self.conv1(x));x=nn.functional.relu(self.conv2(x));x=nn.functional.relu(self.bn1(x))
        x=nn.functional.relu(self.conv3(x));x=nn.functional.relu(self.conv4(x));x=nn.functional.relu(self.bn2(x))
        x=nn.functional.max_pool2d(x,(2,1))
        x=nn.functional.relu(self.conv5(x));x=nn.functional.relu(self.conv6(x));x=nn.functional.relu(self.bn3(x))
        x=nn.functional.max_pool2d(x,(2,1))
        # Keras Flatten uses NHWC ordering.
        return self.fc(x.permute(0,2,3,1).reshape(len(x),-1))

def load_beamsense_teacher(path,device,validation_sample):
    """Load Keras BeamSense weights into a differentiable PyTorch replica."""
    import tensorflow as tf
    try:tf.config.set_visible_devices([],"GPU")
    except RuntimeError:pass
    keras_model=tf.keras.models.load_model(path,compile=False)
    teacher=BeamSenseTeacher()
    keras_conv=[layer for layer in keras_model.layers if layer.__class__.__name__=="Conv2D"]
    keras_bn=[layer for layer in keras_model.layers if layer.__class__.__name__=="BatchNormalization"]
    keras_dense=[layer for layer in keras_model.layers if layer.__class__.__name__=="Dense"]
    torch_conv=[teacher.conv1,teacher.conv2,teacher.conv3,teacher.conv4,teacher.conv5,teacher.conv6]
    torch_bn=[teacher.bn1,teacher.bn2,teacher.bn3]
    if not (len(keras_conv)==6 and len(keras_bn)==3 and len(keras_dense)==1):
        raise RuntimeError("Unexpected BeamSense Keras architecture")
    with torch.no_grad():
        for source,target in zip(keras_conv,torch_conv):
            kernel,bias=source.get_weights();target.weight.copy_(torch.from_numpy(kernel.transpose(3,2,0,1)));target.bias.copy_(torch.from_numpy(bias))
        for source,target in zip(keras_bn,torch_bn):
            gamma,beta,mean,var=source.get_weights();target.weight.copy_(torch.from_numpy(gamma));target.bias.copy_(torch.from_numpy(beta));target.running_mean.copy_(torch.from_numpy(mean));target.running_var.copy_(torch.from_numpy(var))
        kernel,bias=keras_dense[0].get_weights();teacher.fc.weight.copy_(torch.from_numpy(kernel.T));teacher.fc.bias.copy_(torch.from_numpy(bias))
    keras_probability=np.asarray(keras_model(validation_sample,training=False))
    teacher=teacher.to(device).eval().requires_grad_(False)
    with torch.no_grad():
        torch_probability=teacher(torch.from_numpy(validation_sample).to(device)).softmax(1).cpu().numpy()
    maximum_error=float(np.max(np.abs(keras_probability-torch_probability)))
    if maximum_error>1e-3:raise RuntimeError(f"Keras/PyTorch BeamSense mismatch: {maximum_error}")
    print({"teacher_probability_max_error":maximum_error},flush=True)
    return teacher

def sched(n,device,kind='linear'):
    if kind=='linear':
        b=torch.linspace(1e-4,.02,n,device=device)
    elif kind=='cosine':
        s=.008;t=torch.linspace(0,n,n+1,device=device,dtype=torch.float64)
        cumulative=torch.cos(((t/n+s)/(1+s))*math.pi/2).square()
        cumulative=cumulative/cumulative[0]
        b=(1-cumulative[1:]/cumulative[:-1]).clamp(1e-5,.999).float()
    else:raise ValueError(f'unknown diffusion schedule: {kind}')
    a=1-b;ab=torch.cumprod(a,0);return b,a,ab

@torch.no_grad()
def generate(model,y,anchor,n,device,schedule='linear'):
    b,a,ab=sched(n,device,schedule);x=torch.randn(len(y),4,9,234,device=device)
    for i in reversed(range(n)):
        t=torch.full((len(y),),i,dtype=torch.long,device=device);e=model(x,t,y,anchor)
        x=(x-(1-a[i])/torch.sqrt(1-ab[i])*e)/torch.sqrt(a[i])
        if i:x+=torch.sqrt(b[i])*torch.randn_like(x)
    return x.cpu().numpy()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output_dir',type=Path)
    p.add_argument('--split-indices-dir',type=Path);p.add_argument('--split-seed',type=int,default=111)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=20);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--schedule',choices=('linear','cosine'),default='linear',help='cosine reaches near-pure noise and is recommended for short schedules')
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--max-samples',type=int)
    p.add_argument('--generated-per-class',type=int,help='generate this many samples for every activity; scarce anchors are reused with new diffusion noise')
    p.add_argument('--candidates-per-anchor',type=int,default=1,
        help='generate K independent diffusion candidates for every training anchor')
    p.add_argument('--distill-npz',type=Path,
        help='selected one-per-anchor BFA targets to mix 1:1 with real deltas during fine-tuning')
    p.add_argument('--init-checkpoint',type=Path,
        help='initialize model weights from another run without reusing its optimizer or output directory')
    p.add_argument('--teacher-model',type=Path,help='frozen BeamSense Keras checkpoint used for activity guidance')
    p.add_argument('--classification-weight',type=float,default=0.0)
    p.add_argument('--x0-weight',type=float,default=0.0,help='weight for clean normalized-delta reconstruction')
    p.add_argument('--x0-clip',type=float,default=6.0,help='soft bound for predicted x0 used by auxiliary losses')
    p.add_argument('--temporal-weight',type=float,default=0.0,help='weight for per-angle temporal-magnitude matching')
    p.add_argument('--temporal-std-weight',type=float,default=1.0,help='relative weight for channel delta spread within temporal loss')
    p.add_argument('--temporal-tail-weight',type=float,default=0.5,help='relative weight for channel delta p95 within temporal loss')
    p.add_argument('--teacher-normalization',choices=('none','angle-range'),default='none')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.candidates_per_anchor < 1:p.error('--candidates-per-anchor must be >= 1')
    if args.generated_per_class and args.candidates_per_anchor != 1:
        p.error('--generated-per-class and --candidates-per-anchor cannot be combined')
    if args.resume and args.init_checkpoint:p.error('--resume and --init-checkpoint are mutually exclusive')
    if min(args.classification_weight,args.x0_weight,args.temporal_weight,args.temporal_std_weight,args.temporal_tail_weight)<0:p.error('loss weights must be nonnegative')
    if args.x0_clip<=0:p.error('--x0-clip must be positive')
    if args.classification_weight and not args.teacher_model:p.error('--classification-weight requires --teacher-model')
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
    normalized=(raw-mean)/std;labels=y[train];anchors=anchor_features(x[train,:1]);raw_anchors=x[train,0].astype(np.float32)
    distill_samples=0
    if args.distill_npz:
        with np.load(args.distill_npz,allow_pickle=False) as distilled:
            if 'allocation_real_index' not in distilled.files:p.error('--distill-npz lacks allocation_real_index')
            if ('train_split_seed' not in distilled.files
                    or int(distilled['train_split_seed'])!=args.split_seed):
                p.error('--distill-npz was not produced from the requested training split')
            dx=distilled['x'];dy=distilled['y'].astype(np.int64);di=distilled['allocation_real_index'].astype(np.int64)
            if dx.shape[1:]!=(10,234,4):p.error(f'bad distilled BFA shape {dx.shape}')
            if not (len(dx)==len(dy)==len(di)):p.error('distilled x/y/allocation lengths differ')
            if np.any((di<0)|(di>=len(y))):p.error('distilled allocation_real_index is out of range')
            if len(np.unique(di))!=len(di):p.error('--distill-npz must contain exactly one sample per anchor')
            lookup=np.full(len(y),-1,dtype=np.int64);lookup[di]=np.arange(len(di));rows=lookup[train]
            if np.any(rows<0):p.error('--distill-npz does not cover every real training anchor')
            if np.any(dy[rows]!=y[train]):p.error('distilled labels do not match real anchors')
            distilled_normalized=(signed_delta(dx[rows])-mean)/std
        normalized=np.concatenate((normalized,distilled_normalized))
        labels=np.concatenate((labels,y[train]));anchors=np.concatenate((anchors,anchors.copy()));raw_anchors=np.concatenate((raw_anchors,raw_anchors.copy()))
        distill_samples=len(rows)
    ds=TensorDataset(torch.from_numpy(normalized).permute(0,3,1,2),torch.from_numpy(labels),torch.from_numpy(anchors),torch.from_numpy(raw_anchors))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    model=Model().to(device)
    if args.init_checkpoint:
        if not args.init_checkpoint.is_file():p.error(f'missing {args.init_checkpoint}')
        initial=torch.load(args.init_checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(initial['model'])
        if not np.array_equal(initial['train_indices'],train):p.error('initial checkpoint used different train indices')
        if not (np.allclose(initial['mean'],mean) and np.allclose(initial['std'],std)):
            p.error('initial checkpoint normalization statistics differ')
        print(f'Initialized model from {args.init_checkpoint}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate);_,_,ab=sched(args.steps,device,args.schedule)
    teacher=None
    if args.teacher_model:
        teacher=load_beamsense_teacher(args.teacher_model,device,x[train[:2]].astype(np.float32))
    mean_tensor=torch.from_numpy(mean).to(device)[None,None,None,:]
    std_tensor=torch.from_numpy(std).to(device)[None,None,None,:]
    teacher_scale=torch.tensor([511.,511.,127.,127.],device=device)[None,None,None,:]
    args.output_dir.mkdir(parents=True,exist_ok=True);history=[];component_history=[];start=0
    if args.resume:
        s=torch.load(ck,map_location=device,weights_only=False);model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
        if s.get('schedule','linear')!=args.schedule:p.error('diffusion schedule changed while resuming')
        history=s['history'];component_history=s.get('component_history',[]);start=s['epoch']
        if not np.array_equal(s['train_indices'],train):p.error('train indices changed')
        mean,std=s['mean'],s['std'];print(f'Resuming after epoch {start}',flush=True)
    for epoch in range(start,args.epochs):
        losses=[];diff_losses=[];classification_losses=[];x0_losses=[];temporal_losses=[];temporal_mean_losses=[];temporal_spread_losses=[];temporal_tail_losses=[];model.train()
        for clean,label,anchor,raw_anchor in loader:
            clean,label,anchor,raw_anchor=clean.to(device),label.to(device),anchor.to(device),raw_anchor.to(device)
            t=torch.randint(args.steps,(len(clean),),device=device);noise=torch.randn_like(clean);q=ab[t][:,None,None,None]
            noisy=torch.sqrt(q)*clean+torch.sqrt(1-q)*noise;predicted_noise=model(noisy,t,label,anchor)
            diffusion_loss=nn.functional.mse_loss(predicted_noise,noise);loss=diffusion_loss
            predicted_x0=(noisy-torch.sqrt(1-q)*predicted_noise)/torch.sqrt(q.clamp_min(1e-8))
            guided_x0=args.x0_clip*torch.tanh(predicted_x0/args.x0_clip)
            x0_loss=nn.functional.smooth_l1_loss(guided_x0,clean)
            temporal_loss,temporal_mean_loss,temporal_spread_loss,temporal_tail_loss=temporal_fidelity_loss(
                guided_x0,clean,args.temporal_std_weight,args.temporal_tail_weight
            )
            classification_loss=torch.zeros((),device=device)
            if teacher is not None and args.classification_weight:
                raw_delta=guided_x0.permute(0,2,3,1)*std_tensor+mean_tensor
                reconstructed=continuous_reconstruct(raw_anchor,raw_delta)
                teacher_input=reconstructed/teacher_scale if args.teacher_normalization=='angle-range' else reconstructed
                classification_loss=nn.functional.cross_entropy(teacher(teacher_input),label)
            loss=loss+args.x0_weight*x0_loss+args.temporal_weight*temporal_loss+args.classification_weight*classification_loss
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
            diff_losses.append(diffusion_loss.item());classification_losses.append(classification_loss.item());x0_losses.append(x0_loss.item());temporal_losses.append(temporal_loss.item());temporal_mean_losses.append(temporal_mean_loss.item());temporal_spread_losses.append(temporal_spread_loss.item());temporal_tail_losses.append(temporal_tail_loss.item())
        history.append(float(np.mean(losses)));components=dict(diffusion=float(np.mean(diff_losses)),classification=float(np.mean(classification_losses)),x0=float(np.mean(x0_losses)),temporal=float(np.mean(temporal_losses)),temporal_mean=float(np.mean(temporal_mean_losses)),temporal_spread=float(np.mean(temporal_spread_losses)),temporal_tail=float(np.mean(temporal_tail_losses)));component_history.append(components)
        torch.save(dict(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),history=history,component_history=component_history,
            train_indices=train,mean=mean,std=std,schedule=args.schedule,distill_npz=str(args.distill_npz) if args.distill_npz else None),ck)
        print({'epoch':epoch+1,'loss':history[-1],**components},flush=True)
    if args.generated_per_class:
        generation_rng=np.random.default_rng(args.seed+1);generation=[]
        for label in range(20):
            candidates=train[y[train]==label]
            if not len(candidates):p.error(f'no training anchor available for class {label}')
            generation.extend(generation_rng.choice(candidates,args.generated_per_class,
                replace=len(candidates)<args.generated_per_class))
        generation=np.asarray(generation,dtype=np.int64)
    else:generation=np.repeat(train,args.candidates_per_anchor)
    generation_labels=y[generation];generation_anchors=anchor_features(x[generation,:1])
    generation_suffix=(f'_per_class_{args.generated_per_class}' if args.generated_per_class
        else f'_candidates_{args.candidates_per_anchor}' if args.candidates_per_anchor>1 else '')
    chunks=args.output_dir/f'generation_chunks{generation_suffix}';chunks.mkdir(exist_ok=True);made=[]
    for begin in range(0,len(generation),args.batch_size):
        end=min(begin+args.batch_size,len(generation));path=chunks/f'{begin:08d}_{end:08d}.npy'
        if path.is_file():out=np.load(path,allow_pickle=False)
        else:
            label=torch.from_numpy(generation_labels[begin:end]).to(device);anchor=torch.from_numpy(generation_anchors[begin:end]).to(device)
            delta=generate(model,label,anchor,args.steps,device,args.schedule).transpose(0,2,3,1)*std+mean
            out=reconstruct(x[generation[begin:end],0],delta);tmp=path.with_suffix('.tmp.npy');np.save(tmp,out);tmp.replace(path)
            print({'generated':end,'total':len(generation)},flush=True)
        made.append(out)
    generated_path=args.output_dir/f'generated_bfa{generation_suffix}.npz'
    protocol_path=args.output_dir/f'protocol{generation_suffix}.json'
    made=np.concatenate(made);np.savez_compressed(generated_path,x=made,y=generation_labels,
        **{k:v[generation] for k,v in meta.items()},allocation_real_index=generation,augmentation_eligible=np.ones(len(generation),bool),
        candidate_rank=np.tile(np.arange(args.candidates_per_anchor,dtype=np.int16),len(train)) if not args.generated_per_class else np.zeros(len(generation),np.int16),
        train_split_seed=np.asarray(args.split_seed),augmentation=np.asarray('anchor-conditioned-bfa-delta-ddpm-v1'))
    protocol_path.write_text(json.dumps(dict(input=str(args.input),device=str(device),seed=args.seed,
        split_seed=args.split_seed,steps=args.steps,schedule=args.schedule,epochs=args.epochs,train_samples=len(train),
        optimization_samples=len(ds),distill_samples=distill_samples,
        distill_npz=str(args.distill_npz) if args.distill_npz else None,
        teacher_model=str(args.teacher_model) if args.teacher_model else None,
        classification_weight=args.classification_weight,x0_weight=args.x0_weight,x0_clip=args.x0_clip,temporal_weight=args.temporal_weight,
        temporal_std_weight=args.temporal_std_weight,temporal_tail_weight=args.temporal_tail_weight,
        teacher_normalization=args.teacher_normalization,
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,generated_samples=len(generation),
        generated_per_class=args.generated_per_class,candidates_per_anchor=args.candidates_per_anchor,
        generated_class_counts=np.bincount(generation_labels,minlength=20).tolist(),
        unique_generation_anchors=len(np.unique(generation)),delta_mean=mean.tolist(),
        delta_std=std.tolist(),loss=history,loss_components=component_history),indent=2));print('Saved',made.shape,flush=True)

if __name__=='__main__':main()
