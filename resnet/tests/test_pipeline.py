"""Synthetic integration checks, never dataset performance evidence."""
import argparse
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from resnet.prepare_local import prepare, WRAPPERS, BKTree
from resnet.engine import validate_partitions,load_checkpoint
from resnet.train import train
from resnet.evaluate_local import evaluate

class ResNetPipelineTests(unittest.TestCase):
    def test_bktree_matches_bruteforce(self):
        rng=np.random.default_rng(42); values=[int(x) for x in rng.integers(0,2**63,size=100)]
        tree=BKTree()
        for v in values:tree.add(v)
        for q in values[:10]+[0,12345]:
            for radius in [0,1,4,64]:self.assertEqual(tree.contains(q,radius),any((q^v).bit_count()<=radius for v in values))
    def test_overlap_guard(self):
        a=dict(image_id='a',relative_path='a.jpg',duplicate_group='same',label='0',split='train')
        b=dict(image_id='b',relative_path='b.jpg',duplicate_group='b',label='1',split='train')
        c=dict(image_id='c',relative_path='c.jpg',duplicate_group='same',label='0',split='val')
        d=dict(image_id='d',relative_path='d.jpg',duplicate_group='d',label='1',split='val')
        with self.assertRaises(ValueError):validate_partitions([a,b],[c,d])
    def test_prepare_train_resume_evaluate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'images';md=base/'manifests';rng=np.random.default_rng(5201)
            for gen in ['sd14','biggan','adm','midjourney']:
                subsets=[('val','ai',6)]
                if gen=='sd14':subsets += [('val','nature',6),('train','ai',12),('train','nature',12)]
                for split,label,n in subsets:
                    folder=root/gen/WRAPPERS[gen]/split/label;folder.mkdir(parents=True)
                    for i in range(n):Image.fromarray(rng.integers(0,256,size=(80,80,3),dtype=np.uint8)).save(folder/f'{i}.png')
            source=root/'sd14'/WRAPPERS['sd14']/'val/nature/0.png'
            duplicate=root/'sd14'/WRAPPERS['sd14']/'train/nature/duplicate.png';duplicate.write_bytes(source.read_bytes())
            pa=argparse.Namespace(data_root=str(root),manifest_dir=str(md),train_n=8,val_n=4,test_n=2,seed=42,
                                  train_generator='sd14',test_generators=['sd14','biggan','adm','midjourney'],phash_radius=4)
            audit=prepare(pa)
            self.assertEqual(audit['counts']['train']['n'],8);self.assertEqual(audit['counts']['val']['n'],4)
            groups=[]
            for csvfile in md.glob('*.csv'):
                with csvfile.open() as f:groups += [r['duplicate_group'] for r in csv.DictReader(f)]
            self.assertEqual(len(groups),len(set(groups)))
            with self.assertRaises(FileExistsError):prepare(pa)
            config=dict(seed=42,data=dict(root=str(root),train_manifest=str(md/'train.csv'),val_manifest=str(md/'val.csv')),
                        training=dict(batch_size=4,epochs=2,learning_rate=.0001,weight_decay=.0001,workers=0,patience=0),
                        augmentation=dict(enabled=True),paths=dict(output_dir=str(base/'run')))
            conf=base/'config.yaml';conf.write_text(yaml.safe_dump(config))
            ta=argparse.Namespace(config=str(conf),run_dir=None,resume=False,device='cpu',batch=None,workers=0,epochs=1,threads=2,no_pretrained=True)
            out=train(ta);self.assertEqual(load_checkpoint(out/'last.pt')['epoch'],1)
            ta.resume=True;ta.epochs=2;train(ta)
            last=load_checkpoint(out/'last.pt');self.assertEqual(last['epoch'],2);self.assertEqual(len(last['history']),2)
            self.assertTrue((out/'learning_curves.png').is_file());self.assertIn('calibration',load_checkpoint(out/'best.pt'))
            ea=argparse.Namespace(checkpoint=str(out/'best.pt'),manifest_dir=str(md),data_root=None,output=str(base/'test'),device='cpu',batch=4,workers=0)
            results=evaluate(ea);self.assertEqual(len(results),4)
            self.assertTrue(all(r['n']==4 for r in results));self.assertTrue((base/'test/summary.json').exists())
            with self.assertRaises(FileExistsError):evaluate(ea)
            with (md/'train.csv').open('a') as f:f.write('\n')
            with self.assertRaises(ValueError):train(ta)
if __name__=='__main__':unittest.main()
