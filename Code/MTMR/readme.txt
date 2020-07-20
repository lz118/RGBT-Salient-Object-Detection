The code is for paper "Saliency Detection via Graph-Based Manifold Ranking" 
by Chuan Yang, Lihe Zhang, Huchuan Lu, Ming-Hsuan Yang, and Xiang Ruan
To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
written by Chuan Yang
Email: ycscience86@gmail.com
******************************************************************************************************************
The code is tested on Windows XP with MATLAB R2010b.
******************************************************************************************************************
Usage:
>put the test images into file '\test'
>run 'demo.m'
******************************************************************************************************************
Note: We observe that some images on the MSRA dataset are surrounded with artificial frames,
which will invalidate the used boundary prior. Thus, we run a pre-processing to remove such obvious frames.

Procedures:
1. compute a binary edge map of the image using the canny method.
2. if a rectangle is detected in a band of 30 pixels in width along the four sides of the edge map (i.e. we assume that the frame is not wider than 30 pixels), we will cut the aera outside the rectangle from the image.
           
The file 'removeframe.m' is the pre-processing code.

******************************************************************************************************************
We use the SLIC superpixel software to generate superpixels (http://ivrg.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html)
and some graph functions in the Graph Analysis Toolbox (http://eslab.bu.edu/software/graphanalysis/).

Note: The running time reported in our paper does not include the time of the pre-processing and the running time of the superpixel generation is computed by using the SLIC Windows GUI based executable.