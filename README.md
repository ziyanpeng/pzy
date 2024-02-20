# LMSwin_PNet
The code with paper "A method for building extraction in remote sensing images based on SwinTransformer"

Remote sensing image building segmentation, which is essential in land use and urban planning, is evolving with deep learning advancements. Conventional methods using convolutional neural networks face limitations in integrating local and global information and establishing long-range dependencies, resulting in suboptimal segmentation in complex scenarios. In contrast, transformer based networks have gained prominence in computer vision, attributed to their effective self-attention mechanism. This paper proposes LMSwin_PNet, a novel segmentation network. It addresses the Swin Transformer encoder's deficiency in local information processing through a Local Feature Extraction Module. Additionally, it features a multiscale nonparameter attention module to enhance feature channel correlations. The network also incorporates the pyramid large kernel convolution module, substituting traditional 3x3 convolutions in the decoder with more computationally efficient 1x1 convolutions, thereby enabling a large receptive field and detailed information capture. Comparative analyses on two public building datasets demonstrate the model's superior segmentation performance and robustness. The results indicate that LMSwin_PNet produces outputs closely matching labels, showing potential for broader application in remote sensing image segmentation tasks. The source code will be freely available at https://github.com/ziyanpeng/pzy.

LMSwin_ The overall network architecture of PNet：

![network](https://github.com/ziyanpeng/pzy/blob/master/network.png)

Visualization results of segmentation using different (Sota) semantic segmentation networks：

![xiaorong](https://github.com/ziyanpeng/pzy/blob/master/xiaorong.png)
Data Availability Statement: The data used in this study are from open-source datasets. The datasets can be downloaded from Road and Building Detection Datasets (toronto.edu), https://gpcv.whu.edu.cn/data/building_dataset.html and Download – Inria Aerial Image Labeling Dataset(accessed on 17 January 2024).

Our training data has been uploaded to Baidu Cloud: 链接：https://pan.baidu.com/s/1Tq2ft-g4B3_dqRY9N9ytPA?pwd=1234 
提取码：1234

If you have any questions during use, please contact us. Our email address is: m210200619@st.shou.edu.cn.

