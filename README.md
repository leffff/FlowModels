# InstructFlow
The aim of this repository is to create a Flow-Matching-based model capable of doing text2img, solving inverse problems: jpeg restoration, debluring, denoising, superres, coloring, inpainting, outpainting.


## text2img generation example
Rectflow $^1$ full steps
![text2img](images/text2img.png)

Rectflow $^1$ 5 steps
![text2img](images/recflow_1_5_steps.png)

Rectflow $^2$ 5 steps
![text2img](images/recflow_2_5_steps.png)

## Inversion example
Input Image                                             |  FM Inversed                                                 |  Changed prompt
:------------------------------------------------------:|:------------------------------------------------------------:|:---------------------------------------:
<img src="images/puppy.png" alt="drawing" width="350"/> | <img src="images/puppy_rec.png" alt="drawing" width="350"/>  |  <img src="images/kitten.png" alt="drawing" width="350"/>

# TODO:
- [x] [Train a foundint text2img model](sd_2_fm_finetuning.ipynb)
- [x] [FM Inversion](https://github.com/leffff/InstructFlow/blob/main/sd_2_fm_inversion.ipynb)
- [x] [Midpoint, Euler, RK4 solvers (with CFG)](https://github.com/leffff/InstructFlow/blob/main/instructflow/generation.py)
- [x] [N-th Reflow (for straightening and fast simulation)](https://github.com/leffff/InstructFlow/blob/main/sd_2_fm_finetuning_reflow_k_order.ipynb)
- [ ] Rectified Flow Distillation
- [ ] Lattent Adversarial Diffusion Distillaiton (4 steps)
- [ ] Train a model for jpeg restoration, debluring, denoising, superres, coloring
- [ ] Train a model for inpainting, outpainting
- [ ] RLHF
- [ ] Bridge Matching
