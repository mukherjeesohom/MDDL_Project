# MDDL Project: Condensing CNNs with Partial Differential Equations

In this project, we explore the so called Global layer proposed in [Kag and Saligrama [2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kag_Condensing_CNNs_With_Partial_Differential_Equations_CVPR_2022_paper.pdf), which is a PDE-based feature layer aimed as capturing global receptive field without additional overheads of large kernel size or large depth which is common in existing CNN architectures. Such Global layers created by enforcing PDE constraints on feature maps have been suggested to create richer feature maps, and can be embedded in any existing CNN architecture reducing its depth. We conduct extensive experiments comparing Global layer-based CNNs versus existing CNN (across datasets and architectures as well as ablations studies), and arrive at the conclusion that the Global layer can successfully reduce the computational and storage budget of CNNs by a substantial amount at negligible of no loss in performance.

### Requirements
1. Python 3.8
2. TensorFlow 2.x
2. Matplotlib
3. NumPy

## Experiments

### (a) MNIST Illustrative Example
The type of model can be chosen by the argument `--model` between `cnn`, `residual`, and `pde`.
```
cd MNIST
python train.py --model pde
```

### (b) CIFAR-10 Baseline
The type of model can be chosen by the argument `--model` between `resnet32` and `resnet32_global`.
```
cd CIFAR
python train.py --model resnet32_global
```
We can also calculate the number of parameters and MACs for any model by running the script `calc_ops.py` with the desired `model_name`.

### (b) CIFAR-10 Ablations
The CIFAR-1O ablations can be run using the same commands as above for CIFAR-10 baseline, but we need to make certain modifications in the script `global_layer.py` for different settings for each row in Table 2 of the report: 

- For both velocity coeffficient and diffusivity as basic depth-wise convolution (default setup for the baseline experiments Table 2 -- Row 2): `Dxy_mode = 'learnable'`, `custom_uv = 'DwConv'`, `custom_dxy  = 'DwConv'`
- For velocity coeffficient as identity and diffusivity as constant (Table 2 -- Row 3): `Dxy_mode = 'constant'` and `custom_uv = 'identity'`
- For both velocity coeffficient and diffusivity as basic residual block (Table 2 -- Row 4): `Dxy_mode = 'learnable'`, `custom_uv = 'BasicBlock'`, `custom_dxy  = 'BasicBlock'`
- For velocity coeffficient as depth-wise convolution and diffusivity as nonlinear isotropic (Table 2 -- Row 5): `Dxy_mode = 'nonlinear_isotropic'` and `custom_uv = 'DwConv'`
- For both velocity coeffficient and diffusivity as basic depth-wise convolution and initialization as basic residual block (Table 2 -- Row 6): Same as default setup but with `block_type = 'BasicBlock'` in `Line 53`
- For diffusion equation (Table 2 -- Row 7): Use `custom_dxy  = 'DwConv'`. Uncomment `Line 121` `g0 = tf.zeros(h.shape)` and comment out `Line 120` `g0 = h`
- For advection equation (Table 2 -- Row 8): Use `custom_uv = 'DwConv'` and `Dxy_mode = 'advection'`




## Results

### MNIST
| **Backbone**              | **Accuracy (\%)** | **# Params** | **Epochs** |
|---------------------------|-------------------|--------------|------------|
| Conv                      |             64.59 |          526 |        100 |
| Residual                  |             66.04 |          530 |        100 |
| PDE (consttant D)         |             65.31 |          530 |        100 |
| PDE (Nonlinear Isotropic) |             66.24 |          530 |        100 |
| Conv                      |             93.02 |        4,614 |        100 |
| Residual                  |             95.58 |        4,646 |        100 |
| PDE (consttant D)         |              95.8 |        4,646 |        100 |
| PDE (Nonlinear Isotropic) |              95.2 |        4,646 |        100 |

### CIFAR-10
| **Architecture** | **Velocity Coeff. v** |   **Diffusivity D**  | **Accuracy (\%)** |  **Initialization**  |    **# Params**    | **# MACs** |
|:----------------:|:---------------------:|:--------------------:|:-----------------:|:--------------------:|:------------------:|:----------:|
|     ResNet32     |           -           |           -          |       93.92       |       Identity       |        844K        |     75M    |
|  ResNet32-Global |    Depth-wise Conv    |    Depth-wise Conv   |       91.27       |       Identity       |        163K        |     16M    |
|  ResNet32-Global |        Identity       |       Constant       |       90.03       |       Identity       |        158K        |     14M    |
|  ResNet32-Global |     residual_block    | Residual Basic Block |       92.73       |       Identity       |        550K        |     72M    |
|  ResNet32-Global |    Depth-wise Conv    |  Nonlinear Isotropic |       89.79       |       Identity       |        160K        |     16M    |
|  ResNet32-Global |    Depth-wise Conv    |    Depth-wise Conv   |       92.18       | Residual Basic Block |        261K        |     30M    |
|  ResNet32-Global |           0           |    Depth-wise Conv   |       90.06       |       Identity       | Diffusion Equation |            |
|  ResNet32-Global |    Depth-wise Conv    |           0          |       90.21       |       Identity       | Advection Equation |            |


## Contributor
1. [Sohom Mukherjee](https://github.com/mukherjeesohom) (Student Number: 7010515)


## Acknowledgement
We would like to acknowledge the following code repositories on which our code is based:

- [PDE_GlobalLayer](https://github.com/anilkagak2/PDE_GlobalLayer)
- [tensorflow2-cifar](https://github.com/lionelmessi6410/tensorflow2-cifar)
