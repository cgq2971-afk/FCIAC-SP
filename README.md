# FCIAC-SP
Class-adjustable Few-shot Incremental Audio Classification, which can  handle dynamic scenarios where classes can be both added and removed under few-shot conditions.

## Datasets
Three audio datasets, including FSC-89, NSynth-100 and LS-100, are adopted as experimental datasets to evaluate the performance of different methods, which have been widely used in previous works for audio classification. To facilitate reimplementation of the results of this paper, the details of these three audio datasets are described at three websites . What's more, they can be downloaded from the above three websites and be freely used for research purpose.

https://www.modelscope.cn/datasets/pp199124903/LS-100/summary

https://www.modelscope.cn/datasets/pp199124903/FSC-89/summary

https://www.modelscope.cn/datasets/pp199124903/NSynth-100/summary

## Code

```bash
pip install -r requirements.txt
```
Training
```
python train.py -project stdu -dataroot DATAROOT -dataset librispeech -config ./configs/stdu_LS-100_FCIAC -gpu 0
python train.py -project stdu -dataroot DATAROOT -dataset nsynth-100 -config ./configs/stdu_nsynth100_FCIAC.yml -gpu 0
python train.py -project stdu -dataroot DATAROOT -dataset FMC -config ./configs/stdu_fmc89_FCIAC.yml -gpu 0
```
## Contact
Yanxiong Li (eeyxli@scut.edu.cn) and Guoqing Chen (202421012439@scut.edu.cn) School of Electronic and Information Engineering, South China University of Technology, Guangzhou, China
