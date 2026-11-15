# Key numbers (balanced accuracy %, mean over 3 seeds)

## CLIP w/o aug
- SD1.4 (seen): bal.acc 95.6, AUROC 99.2, ECE 1.6, real acc 96.3, AI recall 94.9
- SD1.5: bal.acc 94.8, AUROC 99.0, ECE 2.4, real acc 96.3, AI recall 93.2
- Midjourney: bal.acc 81.3, AUROC 93.7, ECE 13.1, real acc 96.3, AI recall 66.2
- ADM: bal.acc 61.0, AUROC 77.8, ECE 34.4, real acc 96.3, AI recall 25.7
- BigGAN: bal.acc 65.8, AUROC 86.3, ECE 27.5, real acc 96.3, AI recall 35.3
- Unseen avg.: bal.acc 75.7, AUROC 89.2, ECE 19.4, real acc 96.3, AI recall 55.1

## CLIP + aug
- SD1.4 (seen): bal.acc 93.7, AUROC 98.6, ECE 2.3, real acc 93.5, AI recall 94.0
- SD1.5: bal.acc 92.5, AUROC 98.1, ECE 3.4, real acc 93.5, AI recall 91.5
- Midjourney: bal.acc 79.9, AUROC 91.6, ECE 13.0, real acc 93.5, AI recall 66.3
- ADM: bal.acc 62.3, AUROC 77.4, ECE 31.0, real acc 93.5, AI recall 31.2
- BigGAN: bal.acc 62.4, AUROC 83.2, ECE 28.8, real acc 93.5, AI recall 31.4
- Unseen avg.: bal.acc 74.3, AUROC 87.6, ECE 19.0, real acc 93.5, AI recall 55.1

## JPEG q=50 drop (clean -> jpeg50, bal.acc points)
- CLIP w/o aug, SD1.4 (seen): 95.6 -> 69.1 (-26.5)
- CLIP w/o aug, Unseen avg.: 75.7 -> 59.6 (-16.1)
- CLIP + aug, SD1.4 (seen): 93.7 -> 96.3 (+2.6)
- CLIP + aug, Unseen avg.: 74.3 -> 74.3 (+0.0)
