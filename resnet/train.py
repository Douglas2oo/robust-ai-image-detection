"""ResNet-50 training on frozen manifests; validation selection, atomic resume checkpoints."""
import argparse
import copy
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from .model import build_model
from .engine import (resolve,read_rows,sha,validate_partitions,seed_all,loader,metrics,predict_logits,
                     atomic_json,save_checkpoint,load_checkpoint,rng_state,restore_rng,write_predictions)

def plot_history(history,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for split in ['train','val']:
        axes[0].plot([r['epoch'] for r in history],[r[split+'_loss'] for r in history],label=split)
        axes[1].plot([r['epoch'] for r in history],[r[split+'_accuracy'] for r in history],label=split)
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('BCE loss');axes[1].set_ylabel('Accuracy (threshold 0.5)')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)

def calibrate(y,z):
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import roc_curve
    objective=lambda logt: metrics(y,z,temperature=math.exp(logt))['loss']
    opt=minimize_scalar(objective,bounds=(-3.,3.),method='bounded')
    temperature=math.exp(float(opt.x)) if opt.success else 1.
    p=1/(1+np.exp(-np.clip(z/temperature,-700,700)))
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    valid=np.isfinite(thresholds)&(thresholds>=0)&(thresholds<=1)
    idx=np.flatnonzero(valid)
    best=max(idx,key=lambda i:((tpr[i]+1-fpr[i])/2,-abs(thresholds[i]-.5)))
    threshold=float(thresholds[best])
    return {'temperature':temperature,'threshold':threshold,'source':'validation only',
            'temperature_bounds_log':[-3.,3.],'threshold_objective':'validation balanced accuracy; ties nearest 0.5'}

def train(args):
    cfg=yaml.safe_load(resolve(args.config).read_text(encoding='utf-8'))
    required=['seed','data','training','augmentation','paths']
    if any(k not in cfg for k in required):raise ValueError('Incomplete configuration')
    root=resolve(cfg['data']['root']); train_path=resolve(cfg['data']['train_manifest']);val_path=resolve(cfg['data']['val_manifest'])
    train_rows=read_rows(train_path);val_rows=read_rows(val_path);validate_partitions(train_rows,val_rows)
    hashes={'train':sha(train_path),'val':sha(val_path)}
    out=resolve(args.run_dir) if args.run_dir else resolve(cfg['paths']['output_dir'])/f'seed_{cfg["seed"]}'
    if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(f'{out} is not empty; use --resume or a new --run-dir')
    if args.resume and not (out/'last.pt').is_file():raise FileNotFoundError(out/'last.pt')
    device=args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Request a GPU job and check CUDA PyTorch.')
    if args.batch:cfg['training']['batch_size']=args.batch
    workers=args.workers if args.workers is not None else cfg['training'].get('workers',4)
    if args.threads:torch.set_num_threads(args.threads)
    seed_all(cfg['seed']);out.mkdir(parents=True,exist_ok=True)
    ck=load_checkpoint(out/'last.pt') if args.resume else None
    if ck and (ck['config']!=cfg or ck['manifest_sha256']!=hashes or ck['pretrained']!= (not args.no_pretrained)):
        raise ValueError('Resume config/manifests/pretraining changed. Use an independent run directory.')
    model=build_model(pretrained=not args.no_pretrained and not args.resume).to(device)
    lr=float(cfg['training']['learning_rate']);wd=float(cfg['training']['weight_decay'])
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=int(cfg['training']['epochs']))
    amp=device=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];best_loss=float('inf');bad=0;best_epoch=0;start_epoch=0
    if ck:
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        # torch loads optimizer tensors onto the model device in load_state_dict.
        history=ck['history'];best_loss=ck['best_loss'];bad=ck['bad_epochs'];best_epoch=ck['best_epoch'];start_epoch=ck['epoch']
        restore_rng(ck['rng'])
    config_snapshot=copy.deepcopy(cfg)
    atomic_json(out/'config.json',config_snapshot)
    atomic_json(out/'environment.json',dict(torch=torch.__version__,python=platform.python_version(),device=device,
               gpu=torch.cuda.get_device_name(0) if amp else None,pretrained=not args.no_pretrained,manifest_sha256=hashes,
               slurm_job_id=os.getenv('SLURM_JOB_ID'),labels={'real':0,'ai':1},preprocessing='ResNet50 IMAGENET1K_V2 transforms: resize 232, center crop 224, ImageNet normalization'))
    batch=int(cfg['training']['batch_size'])
    td=loader(root,train_rows,batch,workers,True,cfg['augmentation']['enabled'],cfg['seed'])
    vd=loader(root,val_rows,batch,workers)
    max_epochs=min(args.epochs or cfg['training']['epochs'],cfg['training']['epochs'])
    print(f'Device={device} | train={len(train_rows)} val={len(val_rows)} batch={batch} | run={out}',flush=True)
    patience=int(cfg['training'].get('patience',5))
    if not patience or bad<patience:
        for epoch in range(start_epoch,max_epochs):
            started=time.time();model.train();td.dataset.epoch=epoch;loss_sum=0.;correct=0;n=0
            for x,y,_ in tqdm(td,desc=f'Epoch {epoch+1}/{max_epochs}',mininterval=5):
                x=x.to(device,non_blocking=True);y=y.to(device,dtype=torch.float32,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',enabled=amp):
                    z=model(x).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss; checkpoint from previous completed epoch remains available')
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt);scaler.update()
                loss_sum+=float(loss.detach())*len(y);correct+=int(((z.detach()>=0)==(y>=.5)).sum());n+=len(y)
            if n != len(train_rows):raise RuntimeError('Incomplete epoch: expected all training rows')
            yv,zv=predict_logits(model,vd,device);vm=metrics(yv,zv)
            if not np.isfinite(vm['loss']):raise FloatingPointError('Non-finite validation loss')
            row=dict(epoch=epoch+1,train_samples=n,train_loss=loss_sum/n,train_accuracy=correct/n,val_loss=vm['loss'],
                     val_accuracy=vm['accuracy'],val_balanced_accuracy=vm['balanced_accuracy'],val_auroc=vm['auroc'],
                     lr=opt.param_groups[0]['lr'],seconds=time.time()-started)
            history.append(row);scheduler.step()
            common=dict(arch='resnet50',state_dict=model.state_dict(),config=cfg,manifest_sha256=hashes,
                        epoch=epoch+1,pretrained=not args.no_pretrained)
            if vm['loss']<best_loss:
                best_loss=vm['loss'];best_epoch=epoch+1;bad=0;save_checkpoint(out/'best.pt',common)
            else:bad+=1
            save_checkpoint(out/'last.pt',dict(**common,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                              history=history,best_loss=best_loss,best_epoch=best_epoch,bad_epochs=bad,rng=rng_state()))
            atomic_json(out/'history.json',history);plot_history(history,out)
            print('Epoch {epoch}: train_samples={train_samples} train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} val_loss={val_loss:.4f} val_acc={val_accuracy:.2%} val_AUROC={val_auroc:.4f} seconds={seconds:.1f}'.format(**row),flush=True)
            if patience and bad>=patience:
                print('Early stopping on validation loss',flush=True);break
    best=load_checkpoint(out/'best.pt');model.load_state_dict(best['state_dict']);yv,zv=predict_logits(model,vd,device)
    cal=calibrate(yv,zv);best['calibration']=cal;save_checkpoint(out/'best.pt',best)
    atomic_json(out/'calibration.json',cal);atomic_json(out/'validation.json',dict(best_epoch=best['epoch'],raw=metrics(yv,zv),calibrated=metrics(yv,zv,cal['temperature'],cal['threshold'])))
    write_predictions(out/'validation_predictions.csv',val_rows,zv,cal['temperature'],cal['threshold'],out.name)
    print('DONE. Best validation epoch:',best['epoch'],'Checkpoint:',out/'best.pt',flush=True)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='resnet/config.yaml')
    p.add_argument('--run-dir');p.add_argument('--resume',action='store_true');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--batch',type=int);p.add_argument('--workers',type=int);p.add_argument('--epochs',type=int,help='Run through this epoch, bounded by configured schedule')
    p.add_argument('--threads',type=int);p.add_argument('--no-pretrained',action='store_true',help='Debug only: random weights, do not use for formal runs')
    a=p.parse_args()
    if a.epochs is not None and a.epochs<1:p.error('epochs must be positive')
    if a.batch is not None and a.batch<2:p.error('batch must be at least 2')
    train(a)
if __name__=='__main__':main()
