"""ResNet-specific helpers; do not change CLIP's shared implementations."""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
from .manifest import read_manifest

PROJECT=Path(__file__).resolve().parents[1]

def resolve(path):
    p=Path(path).expanduser();return p if p.is_absolute() else PROJECT/p

def read_rows(path):return read_manifest(resolve(path))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)

def save_checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(state,temp);temp.replace(path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if torch.cuda.is_available() and s['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])

def load_checkpoint(path):
    # Load only checkpoints you trained/trust: optimizer/RNG payload uses Python pickle.
    return torch.load(path,map_location='cpu',weights_only=False)

def validate_partitions(train,val):
    for rows,name in [(train,'train'),(val,'val')]:
        if {int(r['label']) for r in rows}!={0,1}:raise ValueError(name+' must contain both labels')
        if any(r['split']!=name for r in rows):raise ValueError('Unexpected split field in '+name)
    for key in ['image_id','relative_path','duplicate_group']:
        a={r[key] for r in train if r[key]};b={r[key] for r in val if r[key]}
        if a&b:raise ValueError('Train/validation overlap in '+key)

class Images(Dataset):
    def __init__(self,root,rows,augment=False,seed=42):
        self.root=Path(root);self.rows=rows;self.augment=augment;self.seed=seed;self.epoch=0
        self.transform=ResNet50_Weights.IMAGENET1K_V2.transforms()
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import io
        r=self.rows[i];path=self.root/r['relative_path']
        with Image.open(path) as raw:im=raw.convert('RGB')
        if self.augment:
            digest=hashlib.sha256(f'{self.seed}:{self.epoch}:{r["image_id"]}'.encode()).digest()
            rng=random.Random(int.from_bytes(digest[:8],'big'))
            if rng.random()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random()<.3:
                buf=io.BytesIO();im.save(buf,format='JPEG',quality=rng.randint(70,95));buf.seek(0)
                with Image.open(buf) as tmp:im=tmp.convert('RGB')
            if rng.random()<.15:im=im.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.8)))
        return self.transform(im),float(r['label']),i

def loader(root,rows,batch,workers,training=False,augment=False,seed=42):
    return DataLoader(Images(root,rows,augment,seed),batch_size=batch,shuffle=training,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),persistent_workers=False)

def metrics(y,logits,temperature=1.,threshold=.5):
    y=np.asarray(y,dtype=int);z=np.asarray(logits,dtype=float)/temperature
    p=1/(1+np.exp(-np.clip(z,-700,700)));pred=(p>=threshold).astype(int)
    loss=float(np.mean(np.logaddexp(0,z)-y*z))
    cm=confusion_matrix(y,pred,labels=[0,1]).tolist()
    return dict(n=len(y),loss=loss,accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
                f1=float(f1_score(y,pred,zero_division=0)),confusion_matrix=cm,threshold=threshold,temperature=temperature)

def predict_logits(model,dl,device):
    model.eval();logits=[];labels=[]
    with torch.inference_mode():
        for x,y,_ in dl:
            z=model(x.to(device,non_blocking=True)).flatten()
            logits.extend(z.float().cpu().tolist());labels.extend(y.tolist())
    return np.asarray(labels,dtype=int),np.asarray(logits)

def write_predictions(path,rows,z,temp,threshold,run_id):
    probs=1/(1+np.exp(-np.clip(np.asarray(z)/temp,-700,700)))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        fields=['run_id','image_id','label','generator','condition','logit','p_ai','prediction']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r,l,p in zip(rows,z,probs):
            w.writerow(dict(run_id=run_id,image_id=r['image_id'],label=r['label'],generator=r['generator'],
                            condition='clean',logit=float(l),p_ai=float(p),prediction=int(p>=threshold)))
