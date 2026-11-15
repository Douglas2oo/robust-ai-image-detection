"""Evaluate an already selected ResNet; pair one fixed real set with each generator."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .model import build_model
from .engine import resolve,read_rows,sha,load_checkpoint,loader,predict_logits,metrics,atomic_json,write_predictions

def evaluate(args):
    ck=load_checkpoint(resolve(args.checkpoint));cfg=ck['config'];root=resolve(args.data_root or cfg['data']['root'])
    if 'calibration' not in ck:raise ValueError('Finish training/resume finalization to calibrate best.pt before evaluation')
    cal=ck['calibration'];temp=cal['temperature'];threshold=cal['threshold']
    md=resolve(args.manifest_dir);real=read_rows(md/'test_real.csv')
    fake_paths=sorted(md.glob('test_*.csv'));fake_paths=[p for p in fake_paths if p.name!='test_real.csv']
    if not fake_paths:raise FileNotFoundError('No test generator manifests')
    train=read_rows(cfg['data']['train_manifest']);val=read_rows(cfg['data']['val_manifest'])
    if ck['manifest_sha256']!={'train':sha(resolve(cfg['data']['train_manifest'])),'val':sha(resolve(cfg['data']['val_manifest']))}:
        raise ValueError('Training/validation manifests changed')
    protected={key:{r[key] for r in train+val if r[key]} for key in ['image_id','relative_path','duplicate_group']}
    if {int(r['label']) for r in real}!={0}:raise ValueError('test_real must be real only')
    groups={};seen_ids=set()
    for path in fake_paths:
        rows=read_rows(path)
        if {int(r['label']) for r in rows}!={1}:raise ValueError('AI test manifest must be fake only')
        if len({r['generator'] for r in rows})!=1:raise ValueError('One generator per fake test manifest')
        groups[rows[0]['generator']]=rows
    all_test=real+[r for rows in groups.values() for r in rows]
    for r in all_test:
        if r['split']!='test':raise ValueError('Unexpected split in test manifest')
        if r['image_id'] in seen_ids:raise ValueError('Duplicate test image ID')
        seen_ids.add(r['image_id'])
        for key,ids in protected.items():
            if r[key] and r[key] in ids:raise ValueError('Test overlaps fitting data in '+key)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU not allocated / CUDA PyTorch unavailable')
    out=resolve(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'test_metrics.json').exists():raise FileExistsError('Use a new output directory to preserve previous test results')
    model=build_model(False).to(args.device);model.load_state_dict(ck['state_dict']);result=[]
    # Infer shared real images only once; their predictions are identical across domain pairings.
    _,zr=predict_logits(model,loader(root,real,args.batch,args.workers),args.device)
    for gen,fake in groups.items():
        _,zf=predict_logits(model,loader(root,fake,args.batch,args.workers),args.device)
        rows=real+fake;z=np.concatenate([zr,zf]);y=[int(r['label']) for r in rows]
        score=metrics(y,z,temp,threshold);score.update(generator=gen,condition='clean')
        result.append(score);write_predictions(out/f'predictions_{gen}.csv',rows,z,temp,threshold,resolve(args.checkpoint).parent.name)
        print(f'{gen}: Accuracy={score["accuracy"]:.2%} BA={score["balanced_accuracy"]:.2%} AUROC={score["auroc"]:.4f}',flush=True)
    atomic_json(out/'test_metrics.json',result)
    macro={key:float(np.mean([r[key] for r in result])) for key in ['accuracy','balanced_accuracy','auroc','f1']}
    macro.update(protocol='Macro across generator pairs sharing the same real images; correlated domain estimates',
                 test_manifest_sha256={p.name:sha(p) for p in [md/'test_real.csv']+fake_paths},
                 checkpoint_sha256=sha(resolve(args.checkpoint)),test_used_for_selection=False)
    atomic_json(out/'summary.json',macro);print('MACRO:',macro,flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest-dir',default='manifests/resnet_v1');p.add_argument('--data-root')
    p.add_argument('--output',default='outputs/resnet_v1/test');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--batch',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    evaluate(p.parse_args())
if __name__=='__main__':main()
