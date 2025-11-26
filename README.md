Gaussian Splatting (Normal + Vector-Quantized)
===========================================================

This repo has my implementation of gaussian splatting along with (work in progress) the vector quantized version. 

**Workflow**: 
1. Run COLMAP SfM on input images → get camera poses + sparse point cloud
2. Initialize Gaussians from COLMAP 3D points
3. Train Gaussians to match input images
4. Render novel views

ATM I have implemented the original version of gaussian splatting and have done testing locally on my M4 mac. Will be doing testing on my PC during the christmas break. Currently over only 1000 iterations on the fern example we get to around PSNR 17. This should improve a lot over more iterations as 3DGS uses ~50000.
