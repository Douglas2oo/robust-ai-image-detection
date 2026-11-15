"""Read-only remote file/access checks. Run in a Slurm CPU job, not on all raw files."""
import argparse,csv,hashlib,json,os
from pathlib import Path

def check(root,manifests,stage,report):
    root=Path(root);md=Path(manifests)
    audit=json.loads((md/'conversion_audit.json').read_text())
    files=[md/'train.csv',md/'val.csv'] if stage=='train' else sorted(md.glob('test_*.csv'))
    seen=set();errors=[];counts={}
    for path in files:
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=audit['converted_manifest_sha256'][path.name]:raise ValueError('Manifest modified: '+str(path))
        with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        counts[path.name]=len(rows)
        for row in rows:
            rel=row['relative_path']
            if rel in seen:continue
            seen.add(rel)
            try:
                with (root/rel).open('rb') as f:
                    if not f.read(1):raise ValueError('Empty file')
            except (OSError,ValueError) as exc:
                errors.append({'relative_path':rel,'error':str(exc)})
            if len(seen)%2000==0:print(f'Checked {len(seen)} images; errors={len(errors)}',flush=True)
    result=dict(stage=stage,data_root=str(root),counts=counts,unique_images=len(seen),errors=errors,
                ok=not errors,scope='Existence/read permission/nonempty only. Does not rehash or decode full image data.')
    dest=Path(report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2),flush=True)
    if errors:
        for error in errors[:10]:print(error,flush=True)
        raise SystemExit('PREFLIGHT_FAILED. See report; do not silently skip source images.')
    print('PREFLIGHT_OK',flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='/project/mscaispod2026/group11')
    p.add_argument('--manifests',default='manifests/clip_shared_v1');p.add_argument('--stage',choices=['train','test'],default='train');p.add_argument('--report',default='outputs/preflight_train.json')
    a=p.parse_args();check(a.data_root,a.manifests,a.stage,a.report)
if __name__=='__main__':main()
