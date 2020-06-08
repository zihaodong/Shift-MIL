# Shift-MIL Pooling and Application
Shift MIL pooling layer based on Noise-And for Classification and Segmentation. It is the implementation of Shift-MIL pooling function based on "Surface Defect Segmentation with Multi-Column patch-wise U-net" by Zihao et al.

## Mathematics Equation and curve 
![](Math_curve.png) 

## Requirements

Tensorflow >= 1.9

Keras >= 2.2

Numpy

## Training and evaluation 

**(1) Classification on MNIST:** python shift_mil.py 

**(2)Segmentation on DAGM 2007 (Weakly-surpvised Learning):** 

**[Data Processing]**

python prepare_datasets_DAGM.py

**[Training stage]**

python run_training.py

**[Test stage]**

python run_testing.py

**[Evaluation stage]**

python eval_curve.py

## Illustration

DAGM 2007 dataset is processed and we will release it to another platform; the results of this project were evaluated on only 63 images from the testset.

## Citation
If you find Shift-MIL pooling application is useful in your research, please consider to cite:

	@inproceedings{chen2019learning,
	  title={Learning Active Contour Models for Medical Image Segmentation},
	  author={Chen, Xu and Williams, Bryan M and Vallabhaneni, Srinivasa R and Czanner, Gabriela and Williams, Rachel and Zheng, Yalin},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  pages={11632--11640},
	  year={2019}
	}

## Other Re-implementation
...
