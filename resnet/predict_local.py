"""Predict one image with this package's calibrated best.pt."""
import argparse,json
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights
from .model import build_model
from .engine import load_checkpoint,resolve

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image');p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    ck=load_checkpoint(resolve(a.checkpoint));cal=ck['calibration'];model=build_model(False).to(a.device)
    model.load_state_dict(ck['state_dict']);model.eval()
    with Image.open(a.image) as raw:x=ResNet50_Weights.IMAGENET1K_V2.transforms()(raw.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():prob=float(torch.sigmoid(model(x.to(a.device)).flatten()[0]/cal['temperature']))
    print(json.dumps(dict(p_ai=prob,p_real=1-prob,prediction='AI' if prob>=cal['threshold'] else 'Real',threshold=cal['threshold']),indent=2))
if __name__=='__main__':main()
