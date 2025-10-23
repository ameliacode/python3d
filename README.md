## Setup

```bash
conda create -n python3d
codna activate python3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && python setup.py install 
cd ..
cd python3d  && pip install -r requirements.txt
```
- [issues](https://github.com/facebookresearch/pytorch3d/issues/1889)
