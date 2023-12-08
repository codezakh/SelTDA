[![Conference](https://img.shields.io/badge/CVPR-2023-blue)](https://openaccess.thecvf.com/content/CVPR2023/html/Khan_Q_How_To_Specialize_Large_Vision-Language_Models_to_Data-Scarce_VQA_CVPR_2023_paper.html)
[![Paper](http://img.shields.io/badge/paper-arxiv.2306.03932-B31B1B.svg)](https://arxiv.org/abs/2306.03932)

**Note:** Code release is in progress. All code has been uploaded, but I'm still working on the documentation.

# SelTDA
This repository will hold the official code of SelTDA, the self-training framework introduced in our CVPR 2023 paper "Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!".


![seltda_teaser](https://user-images.githubusercontent.com/4918041/225918833-7d744775-260a-4bc3-a642-7279531b5b07.png)

## Environment
```bash
conda env create -f environment.yaml
```

## Data
### Downloads and Preprocessing
- [PathVQA](https://github.com/UCSD-AI4H/PathVQA)
    - then use `convert_pathvqa.py`
- [RSVQA](https://rsvqa.sylvainlobry.com/)
    - then use `convert_rsvqa.py`
- OK-VQA and A-OKVQA (use [LAVIS](https://github.com/salesforce/LAVIS))
    - LAVIS should automatically put them in the correct format, but if not, you can use `convert_okvqa.py`
- [VQA Counterexamples](https://github.com/cdancette/detect-shortcuts)
    - then use `convert_vqa_ce.py`
- [AdVQA](https://adversarialvqa.org/download.html)
    - then use `convert_advqa.py`
- [VQA Rephrasings](https://facebookresearch.github.io/VQA-Rephrasings/)
    - then use `convert_vqa_rephrasings.py`

In general, the code expects that each VQA dataset is represented by a single JSON object that is a list of dictionaries. In `schemas.py`, we provide Pydantic models which you can use to define your own datasets or verify that the data is in the correct format. 

## Experiments
See the `examples/` directory to see examples of:
- training the teacher 
    - `examples/train_teacher.sh`
- generating synthetic data with the teacher
    - `examples/generate_synthetic_data.sh`
- self-training with the synthetic data
    - `examples/self_train_synthetic.sh`
- evaluations
    - `examples/evaluate.sh`

## Citation
```
@InProceedings{Khan_2023_CVPR,
    author    = {Khan, Zaid and BG, Vijay Kumar and Schulter, Samuel and Yu, Xiang and Fu, Yun and Chandraker, Manmohan},
    title     = {Q: How To Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15005-15015}
}
```


## Acknowledgements
This code is heavily based on [salesforce/BLIP](https://github.com/salesforce/BLIP).