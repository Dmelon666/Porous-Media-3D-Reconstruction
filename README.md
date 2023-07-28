# Porous-Media-3D-Reconstruction

Python implementation code for the paper titled,

Title: High-Resolution Porous Media 3D Reconstruction Method based on Improved VQGAN

Authors: Zhao Yan

# Installation requirements:

python >= 3.7.0

pytorch >= 1.7.1 + cuda110  

torchvision >= 0.8.1+cu110

To run this code, at least one NVIDIA GeForce RTX3080Ti Super GPU video card with 16GB of video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

# How to use this code:

(1) First phase

1. Preprocessing images.

2. Running the training_vqgan.py file and perform hyperparameter configuration.

(2) Second phase

1. Connect the parameters stored in the first stage.

2. Running the training_transformer.py file and perform hyperparameter configuration.

Finally, the reconstructed 3D Porous medium image is obtained from the set save file for later analysis and processing.

# Reproducing Paper Results:

Note that this repo is primarily designed for simplicity and extending off of our method. Reproducing the full paper results can be done using code found at a separate repo. However, be aware that the code is not as clean.If you have any question, you can contact me via email 840907667@qq.com.


