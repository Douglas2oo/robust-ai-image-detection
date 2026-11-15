# Validation

Base repository commit: e58b40ceceb95ff7033fb3572c1f46847da499d2.

- `python -m unittest discover -s resnet/tests -v`: 4 tests passed (13.744 seconds), CPU PyTorch 2.7.1+cpu / torchvision 0.22.1+cpu / Python 3.12.
- Includes synthetic ResNet training, resume, calibration, evaluation and overlap/tampering guards; synthetic scores are not dataset performance evidence.
- Real frozen team source CSV hashes and converted CSV hashes exactly match the package used for the completed 28,400-image HPC experiment.
- Python compile checks, shell syntax checks and git whitespace checks passed.
- HPC Slurm submission and the actual trained checkpoint were not executed/retrieved in this checkout. Recorded dataset scores in README are from the user's completed HPC job log.
- Tracked changes are confined to resnet/.
