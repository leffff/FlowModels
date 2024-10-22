# Flow Models
The aim of this repository is to test and implement Flow-Matching-based models


## text2img generation example
Rectflow $^1$ full steps
![text2img](images/text2img.png)

Rectflow $^1$ 5 steps
![text2img](images/recflow_1_5_steps.png)

Rectflow $^1$ 5 steps
![text2img](images/rectflow_1_1_step.png)

Rectflow $^2$ 1 step
![text2img](images/recflow_2_5_steps.png)

Rectflow $^2$ 1 step
![text2img](images/rectflow_2_1_step.png)

Rectflow $^2$ + Distill 1 step
![text2img](images/rectflow_2_distill_1step.png)

Rectflow $^2$ + Distill + Midpoint 1 step
![text2img](images/rectflow_2_distill_1_step_midpoint.png)

## Inversion example
Input Image                                             |  FM Inversed                                                 |  Changed prompt
:------------------------------------------------------:|:------------------------------------------------------------:|:---------------------------------------:
<img src="images/puppy.png" alt="drawing" width="350"/> | <img src="images/puppy_rec.png" alt="drawing" width="350"/>  |  <img src="images/kitten.png" alt="drawing" width="350"/>

# TODO:
- [x] [Train a foundint text2img model](sd_2_fm_finetuning.ipynb)
- [x] [FM Inversion](https://github.com/leffff/InstructFlow/blob/main/sd_2_fm_inversion.ipynb)
- [x] [Midpoint, Euler, RK4 solvers (with CFG)](https://github.com/leffff/InstructFlow/blob/main/instructflow/generation.py)
- [x] [N-th Reflow (for straightening and fast simulation)](https://github.com/leffff/InstructFlow/blob/main/sd_2_fm_finetuning_reflow_k.ipynb)
- [x] [Rectified Flow Distillation (1 step)](https://github.com/leffff/InstructFlow/blob/main/sd_2_fm_finetuning_reflow_distil_k.py)
- [x] [Immiscible Diffusion](https://github.com/leffff/FlowModels/blob/main/sd_2_fm_finetuning_immiscible.ipynb)
- [ ] Lattent Adversarial Diffusion Distillaiton (1 step)
- [ ] Diffusion RLHF
- [ ] Bridge Matching
- [ ] Train a model for jpeg restoration, debluring, denoising, superres, coloring
- [ ] Train a model for inpainting, outpainting
