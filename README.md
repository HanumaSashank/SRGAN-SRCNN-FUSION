# Single Image Super Resolution using Two-Stage Neural Network Architecture
## Overview
This project explores enhancing low-resolution images using a two-stage neural network architecture. By fusing the outputs of SRCNN (Super-Resolution Convolutional Neural Network) and SRGAN (Super-Resolution Generative Adversarial Network) models through nine fusion techniques, we achieve superior resolution outcomes. This approach leverages the complementary strengths of both models, yielding highly accurate and visually refined super-resolution images.
## Features
- Implements two leading super-resolution models (SRCNN, SRGAN).
- Experiments with nine innovative fusion techniques, such as Pixel Average Fusion, IHS Fusion, and PCA Fusion.
- Evaluates the fused outputs using industry-standard metrics: PSNR, MSE, and SSIM.
- Provides robust results showing improved resolution quality through fusion.
## Methodology
- Models
  - **SRCNN:** Employs convolution layers for upscaling low-resolution images to higher dimensions.
  - **SRGAN:** Utilizes generative adversarial architecture with a generator and discriminator for photorealistic output.
- Fusion Techniques
  - Pixel Average Fusion
  - Laplacian Pyramid Fusion
  - Principal Component Analysis Fusion (PCA)
  - Feature Level Fusion
  - Region-Wise Fusion
  - Guided Filter Fusion
  - Intensity Hue Saturation (IHS) Fusion
  - Discrete Cosine Transform (DCT) Fusion
  - Select Better Pixel Fusion
- Dataset
  - Training: HuggingFace’s DIV2K dataset.
  - Testing: SET5 images for qualitative and quantitative evaluation.
- Metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - MSE (Mean Squared Error)
  - SSIM (Structural Similarity Index Measure)
## Results
- **Fusion Performance:** IHS Fusion delivered the best results, with superior PSNR, MSE, and SSIM values compared to standalone SRCNN and SRGAN models.
- **Metrics:** Scores for fused outputs (e.g., [33.06 PSNR, 96.3976 MSE, 0.9005 SSIM]) indicated a marked improvement over individual model outputs.
## Tecnologies Used
- **Programming Language:** Python
- **Deep Learning Frameworks:** PyTorch
- **Datasets:** DIV2K, SET5
## Contributions
- **Rohith Reddy Mada:** Implemented SRGAN.
- **Hanuma Shashank Samudrala:** Developed SRCNN.
- **Ashish Athimamula:** Implemented image fusion techniques.
### For more details refer to the full [project report.](https://github.com/user-attachments/files/18513952/CV_PROJECT_REPORT.1.pdf)

