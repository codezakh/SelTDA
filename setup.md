```
conda create -n blip python=3.8
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install timm==0.4.12
pip3 install transformers==4.15.0
pip3 install fairscale==0.4.4
pip3 install pycocoevalcap
pip3 install jupyter
pip3 install ipdb
pip3 install hydra-core
pip3 install ruamel.yaml
pip3 install opencv-python
```

# For development
```
pip3 install pytest
pip3 install pyinstrument
```