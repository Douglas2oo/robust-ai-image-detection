"""Lossless adaptation of frozen team CSVs to the ResNet/common manifest interface."""
import argparse,csv,hashlib,io,json,re,tarfile
from pathlib import Path,PurePosixPath

COLUMNS=['image_id','relative_path','label','generator','semantic_class','split','duplicate_group']
SOURCES=['sd14','sd15','adm','midjourney','biggan']
NAMES=['train.csv','val.csv']+['test_'+s+'.csv' for s in SOURCES]

def digest(data):return hashlib.sha256(data).hexdigest()

def convert(archive,destination):
    dest=Path(destination)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(f'Refusing to overwrite frozen manifests: {dest}')
    files={};original={};converted={}
    from contextlib import nullcontext
    source_dir=Path(archive)
    context=nullcontext(None) if source_dir.is_dir() else tarfile.open(archive,'r:gz')
    with context as t:
        members={m.name.removeprefix('./'):m for m in t.getmembers() if m.isfile()} if t else {}
        for name in NAMES:
            payload=(source_dir/name).read_bytes() if t is None else t.extractfile(members[name]).read();files[name]=digest(payload)
            rows=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))));original[name]=rows
            out=[];seen=set()
            for r in rows:
                rel=r['relpath'];path=PurePosixPath(rel)
                if path.is_absolute() or '..' in path.parts or '\\' in rel or not rel.startswith('genimage/'):
                    raise ValueError('Unsafe/unsupported relative path: '+rel)
                if rel in seen:raise ValueError('Duplicate image path in '+name+': '+rel)
                seen.add(rel)
                label=int(r['label']);source=r['source'];split=r['split'];phash=r['phash'].lower()
                if label not in (0,1) or (label==0)!=(source=='real'):raise ValueError('Inconsistent label/source')
                expected='test' if name.startswith('test_') else name.removesuffix('.csv')
                if split!=expected:raise ValueError('Unexpected split field')
                if not re.fullmatch('[0-9a-f]{16}',phash):raise ValueError('Missing/invalid supplied pHash')
                out.append(dict(image_id=digest(rel.encode()),relative_path=rel,label=label,generator=source,
                                semantic_class=r.get('cls',''),split=split,duplicate_group='supplied_phash:'+phash))
            converted[name]=out
    expected={'train.csv':28400,'val.csv':4000,**{f'test_{s}.csv':2000 for s in SOURCES}}
    for name,count in expected.items():
        rs=converted[name]
        if len(rs)!=count:raise ValueError(f'{name}: expected {count}, found {len(rs)}')
        if sum(r['label']==0 for r in rs)!=count//2:raise ValueError('Expected balanced labels in '+name)
        if name in ['train.csv','val.csv'] and {r['generator'] for r in rs}!={'real','sd14'}:
            raise ValueError('Training/validation must retain real + sd14')
    train={r['relative_path'] for r in converted['train.csv']};val={r['relative_path'] for r in converted['val.csv']}
    test={r['relative_path'] for name,rs in converted.items() if name.startswith('test_') for r in rs}
    if train&val or train&test or val&test:raise ValueError('Source manifests have cross-split path overlap')
    ph=lambda rs:{r['duplicate_group'] for r in rs}
    a=ph(converted['train.csv']);b=ph(converted['val.csv']);c=set().union(*(ph(converted[n]) for n in NAMES if n.startswith('test_')))
    if a&b or a&c or b&c:raise ValueError('Supplied pHash overlaps between fitting and held-out splits')
    real=[r for r in converted['test_sd14.csv'] if r['label']==0];real_identity={r['relative_path']:r for r in real}
    if len(real)!=1000:raise ValueError('Expected 1000 shared test real images')
    outputs={'train.csv':converted['train.csv'],'val.csv':converted['val.csv'],'test_real.csv':real}
    for gen in SOURCES:
        rows=converted['test_'+gen+'.csv'];other={r['relative_path']:r for r in rows if r['label']==0}
        if other!=real_identity:raise ValueError('The test domains do not share exactly the same real images')
        fake=[r for r in rows if r['label']==1]
        if {r['generator'] for r in fake}!={gen}:raise ValueError('Wrong generator in source test CSV')
        outputs['test_'+gen+'.csv']=fake
        if {r['relative_path'] for r in real+fake}!={r['relpath'] for r in original['test_'+gen+'.csv']}:
            raise ValueError('Adapter changed test membership')
    dest.mkdir(parents=True,exist_ok=True);hashes={}
    for name,rows in outputs.items():
        p=dest/name
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
        hashes[name]=digest(p.read_bytes())
    audit=dict(source_archive_sha256=None if source_dir.is_dir() else digest(source_dir.read_bytes()),source_manifest_sha256=files,
               converted_manifest_sha256=hashes,counts={n:len(rs) for n,rs in outputs.items()},unique_test_images=len(test),
               data_root='/project/mscaispod2026/group11',train_generators=['sd14'],test_generators=SOURCES,
               operation='Format conversion only: preserve every source path, label and partition; no sampling, deletion, augmentation or re-splitting',
               identity_note='image_id is SHA256 of relative path, not image content. duplicate_group uses the supplied pHash, not a newly verified pixel hash.',
               checks={'cross_split_relative_path_overlap':0,'cross_split_identical_supplied_phash':0,'shared_real_test_sets_identical':True},
               limitations=['Image bytes and remote permissions not checked locally; run preflight on HPC.',
                            'Source report describes pHash screening. This conversion does not independently establish complete semantic independence.'])
    (dest/'conversion_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'counts':audit['counts'],'unique_test_images':len(test),'checks':audit['checks']},indent=2))
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source','--archive',dest='archive',default='manifests/splits_v1');p.add_argument('--output',default='manifests/clip_shared_v1')
    a=p.parse_args();convert(a.archive,a.output)
if __name__=='__main__':main()
