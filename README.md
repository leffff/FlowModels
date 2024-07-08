# InstructFlow
The aim of this repository is to create a Flow-Matching-based model capable of doing text2img, solving inverse problems: jpeg restoration, debluring, denoising, superres, coloring, inpainting, outpainting.


## text2img generation example
![text2img](images/text2img.png)

## Inversion example
Input Image                                             |  FM Inversed                                                 |  Changed prompt
:------------------------------------------------------:|:------------------------------------------------------------:|:---------------------------------------:
<img src="images/puppy.png" alt="drawing" width="350"/> | <img src="images/puppy_rec.png" alt="drawing" width="350"/>  |  <img src="images/kitten.png" alt="drawing" width="350"/>

# TODO:
- [x] Train a foundint text2img model
- [x] FM Inversion
- [x] Midpoint, Euler, RK4 solvers (with CFG)
- [ ] N-th Reflow (for straightening and fast simulation)
- [ ] Rectified Flow Distillation
- [ ] Lattent Adversarial Diffusion Distillaiton (4 steps)
- [ ] Train a model for jpeg restoration, debluring, denoising, superres, coloring
- [ ] Train a model for inpainting, outpainting
- [ ] RLHF
- [ ] Bridge Matching
