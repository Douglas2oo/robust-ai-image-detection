"""Read-only sampling of local GenImage, with frozen CSV manifests and hash audit."""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image
import imagehash

COLUMNS = ['image_id','relative_path','label','generator','semantic_class','split','duplicate_group']
WRAPPERS = {'sd14':'imagenet_ai_0419_sdv4', 'sd15':'imagenet_ai_0424_sdv5',
            'biggan':'imagenet_ai_0419_biggan', 'adm':'imagenet_ai_0508_adm',
            'midjourney':'imagenet_midjourney'}
EXTS = {'.jpg','.jpeg','.png','.webp','.bmp'}

class BKTree:
    def __init__(self): self.root = None
    def contains(self, value, radius):
        stack = [] if self.root is None else [self.root]
        while stack:
            node = stack.pop(); d = (value ^ node[0]).bit_count()
            if d <= radius: return True
            stack.extend(child for edge,child in node[1].items() if d-radius <= edge <= d+radius)
        return False
    def add(self, value):
        if self.root is None: self.root = [value,{}]; return
        node = self.root
        while True:
            d = (value ^ node[0]).bit_count()
            if d not in node[1]: node[1][d] = [value,{}]; return
            node = node[1][d]

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def candidates(root, source, official_split, label):
    folder = root/source/WRAPPERS[source]/official_split/('nature' if label == 0 else 'ai')
    if not folder.is_dir() or not os.access(folder,os.R_OK|os.X_OK):
        raise PermissionError(f'Cannot read expected image folder: {folder}')
    def fail(error): raise error
    out=[]
    for parent, dirs, names in os.walk(folder, onerror=fail):
        dirs.sort()
        out.extend(Path(parent)/name for name in sorted(names) if Path(name).suffix.lower() in EXTS)
    if not out: raise ValueError(f'No images under {folder}')
    print(f'Indexed {source}/{official_split}/{label}: {len(out)} images',flush=True)
    return out

def prepare(args):
    root = Path(args.data_root).resolve(); dest = Path(args.manifest_dir).resolve()
    # Avoid writing any sampling artifacts into the shared raw-data tree.
    if dest == root or root in dest.parents: raise ValueError('Keep manifests outside the raw dataset')
    if dest.exists() and any(dest.iterdir()): raise FileExistsError(f'Nonempty manifest directory: {dest}; choose a new version')
    if args.train_n <= 0 or args.val_n <= 0 or args.test_n <= 0 or args.train_n%2 or args.val_n%2:
        raise ValueError('Train and validation counts must be positive even numbers; test-n positive')
    if args.train_generator not in args.test_generators: raise ValueError('Include training generator in test generators')
    if len(set(args.test_generators)) != len(args.test_generators): raise ValueError('Duplicate test generators')
    tree=BKTree(); seen=set(); rejected=[]; hashes={}; all_records=[]
    def pick(source, official, label, count, split):
        pool=candidates(root,source,official,label)
        # Independent RNG for each pool; reproducible across machines and Python hash seeds.
        rng=random.Random(f'{args.seed}:{source}:{official}:{label}'); rng.shuffle(pool)
        selected=[]
        for path in pool:
            rel=path.relative_to(root).as_posix()
            try:
                with Image.open(path) as raw:
                    im=raw.convert('RGB'); width,height=im.size
                    digest=hashlib.sha256(f'{width},{height}:'.encode()+im.tobytes()).hexdigest()
                    ph=int(str(imagehash.phash(im)),16); fmt=raw.format
            except PermissionError: raise
            except (OSError,ValueError) as e:
                rejected.append({'path':rel,'reason':'decode_error','detail':str(e)}); continue
            reason='exact_duplicate' if digest in seen else ('near_duplicate' if tree.contains(ph,args.phash_radius) else '')
            if reason:
                rejected.append({'path':rel,'reason':reason}); continue
            seen.add(digest); tree.add(ph)
            ident=hashlib.sha256(rel.encode()).hexdigest()
            record=dict(image_id=ident,relative_path=rel,label=label,generator='real' if label==0 else source,
                        semantic_class='',split=split,duplicate_group=digest)
            hashes[ident]=dict(decoded_sha256=digest,phash=f'{ph:016x}',width=width,height=height,format=fmt,
                               source=source,official_split=official)
            selected.append(record); all_records.append(record)
            if len(selected)%100==0: print(f'Accepted {source}/{official}/{label}: {len(selected)}/{count}',flush=True)
            if len(selected)==count: return selected
        raise ValueError(f'Insufficient unique readable images: {source}/{official}/{label}: {len(selected)}/{count}')
    # Official val is locked test; test images always get priority during global deduplication.
    manifests={'test_real':pick(args.train_generator,'val',0,args.test_n,'test')}
    for gen in args.test_generators: manifests['test_'+gen]=pick(gen,'val',1,args.test_n,'test')
    nr=args.train_n//2; nv=args.val_n//2
    real=pick(args.train_generator,'train',0,nr+nv,'train')
    fake=pick(args.train_generator,'train',1,nr+nv,'train')
    rng=random.Random(args.seed); rng.shuffle(real); rng.shuffle(fake)
    manifests['val']=real[:nv]+fake[:nv]; manifests['train']=real[nv:]+fake[nv:]
    for name in ['train','val']:
        for record in manifests[name]: record['split']=name
    # Write only after successful selection; never partially overwrite a frozen split.
    dest.mkdir(parents=True,exist_ok=True); manifest_hashes={}
    for name,rows in manifests.items():
        rows.sort(key=lambda r:r['image_id']); path=dest/(name+'.csv')
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
        manifest_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    audit=dict(data_root=str(root),seed=args.seed,train_generator=args.train_generator,test_generators=args.test_generators,
               counts={k:dict(n=len(v),labels=dict(Counter(str(r['label']) for r in v))) for k,v in manifests.items()},
               manifest_sha256=manifest_hashes,phash_radius=args.phash_radius,rejected=rejected,image_hashes=hashes,
               semantic_protocol='Unrestricted classes; no inferred numeric-ID to WordNet mapping. Not a class-matched subset.',
               dedup_scope='Selected samples only; exact decoded RGB and global pHash screening. Near duplicates are heuristically excluded.',
               excluded_known_sources=[g for g in WRAPPERS if g not in args.test_generators])
    atomic_json(dest/'audit.json',audit)
    print(json.dumps(audit['counts'],indent=2),flush=True)
    print('Frozen manifests saved:',dest,flush=True)
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--manifest-dir',default='manifests/resnet_v1')
    p.add_argument('--train-generator',choices=list(WRAPPERS),default='sd14')
    p.add_argument('--test-generators',nargs='+',choices=list(WRAPPERS),default=['sd14','biggan','adm','midjourney'])
    p.add_argument('--train-n',type=int,default=8000);p.add_argument('--val-n',type=int,default=2000)
    p.add_argument('--test-n',type=int,default=500);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--phash-radius',type=int,default=4)
    a=p.parse_args()
    if not 0 <= a.phash_radius <= 64:p.error('phash-radius must be between 0 and 64')
    prepare(a)
if __name__=='__main__':main()
