# WavePaint

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavepaint-resource-efficient-token-mixer-for/image-inpainting-on-imagenet)](https://paperswithcode.com/sota/image-inpainting-on-imagenet?p=wavepaint-resource-efficient-token-mixer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavepaint-resource-efficient-token-mixer-for/image-inpainting-on-celeba-hq)](https://paperswithcode.com/sota/image-inpainting-on-celeba-hq?p=wavepaint-resource-efficient-token-mixer-for)

## Resource-efficient Token-mixer for Self-supervised Inpainting

[[arXiv](https://arxiv.org/abs/2307.00407v1)]

![7rklek](https://github.com/pranavphoenix/WavePaint/assets/15833382/c3d2d6de-ebf8-430b-8f76-bb9fc369faac)

Thick Mask

![7rkp3w](https://github.com/pranavphoenix/WavePaint/assets/15833382/48b907b1-e1cc-417d-847d-ea5aab0bec9a)

Medium Mask

![7rkxzr](https://github.com/pranavphoenix/WavePaint/assets/15833382/55d4e3aa-b132-4323-8b5b-792cc63d1069)

Thin Mask

Model Architecture

![image](https://github.com/pranavphoenix/WavePaint/assets/15833382/5f414f26-44f7-4a90-83d8-a35500e21f20)


### Training using train.py

Change the path to input directory containing images. Set the output image size as required. Modify the model parameters and training configurations. Use can use either medium or thick masks.

```bash
python train.py -batch <batch-size> -mask <mask-size>
```

### Running Inference using infer.py
Provide the path to save model pth file, folder containing validation images ground truth and masks, folder to which model outputs to be saved, folder to which masked images to be saved 

```bash
python infer.py 
```

### Calculating performance metrics using evaluate.py
Provide the path to save model pth file, folder containing validation images ground truth and masks, folder to which model outputs to be saved, folder to which masked images to be saved 

```bash
evaluate.py <path/to/Ground/truth/images> <path/to/model/output> <path/to/save/metrics.csv>
```

We have used LaMa training and inference codes for our experiments from https://github.com/advimman/lama



## Citation
If you found this code helpful, please consider citing: 
```
@misc{jeevan2023wavepaint,
      title={WavePaint: Resource-efficient Token-mixer for Self-supervised Inpainting}, 
      author={Pranav Jeevan and Dharshan Sampath Kumar and Amit Sethi},
      year={2023},
      eprint={2307.00407},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
