# DFQF: Data free quantization-aware Fine-tuning

## Requirements
- python 3
- pytorch 
- torchvision

## RUN
First, you should train a teacher network.

```shell
python train_teacher.py
```

If you want to run data-free knowledge distillation application
```shell
python dfkd.py --n_epochs 2000
```
You can use --p_is --p_adv --p_bn to adjust the hyper-parameters for each loss 

If you want to run data free quantization-aware fine-tuning application, first warm-up the generator, then fine-tune the quantized model.
```shell
python dfkd.py 
python dfqf.py
```
You can use --p_wbit and --p_abit to change the quantized bits 

For simplicity, we do not update the generator in fine-tune stage.
