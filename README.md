# DeepSegThoracic
Benchmark UNet-like deep learning architectures for abdominal segmentation

This is the official repository for the paper titled "[Architecture Analysis and Benchmarking of 3D U-Shaped Deep Learning Models for Thoracic Anatomical Segmentation](https://ieeexplore.ieee.org/abstract/document/10669568/references#references)" by Arash Harirpoush, Amirhossein Rasoulian, Marta Kersten-Oertel, and Yiming Xiao. The repository includes the source code and model weights that were presented in the paper. You can also access the trainer for each model, which has been implemented in the nnunet framework.

To use this repository, please cite the following:

Arash Harirpoush, Amirhossein Rasoulian, Marta Kersten-Oertel, and Yiming Xiao. "Architecture Analysis and Benchmarking of 3D U-shaped Deep Learning Models for Thoracic Anatomical Segmentation." IEEE Access (2024).

<img src="https://github.com/HealthX-Lab/DeepSegAbdominal/blob/main/Assets/Labels.png" alt="Labels">

## Overview
[Model Architectures](https://github.com/HealthX-Lab/DeepSegAbdominal/tree/main/nnunet/network_architecture):
  - The folder contains the implementation of the deep learning models used in the study.

[Model Weights](https://github.com/HealthX-Lab/DeepSegAbdominal/tree/main/nnUNet_trained_models):
   -  The folder containing the weight of the trained models.

[Trainers](https://github.com/HealthX-Lab/DeepSegAbdominal/tree/main/nnunet/training/network_training/nnUNet_variants/architectural_variants):
   -  Trainer of  each model in the nnunet framework.

## Getting Started

### Training

#### Preparing the dataset
To train our models effectively, follow these steps to set up the dataset:
1. **Combine the Labels:**
- Use the `combine_masks` and `combine_masks_to_multilabel_file` functions in [libs](https://github.com/HealthX-Lab/DeepSegAbdominal/blob/main/Preprocessing/libs.py) file to combine labels and generate the multi-label map of the labels based on the totalsegmentator implementation.
2. **Dataset Preprocessing:**
- Preprocess the dataset based on the [nnunet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) instructions.
3. **Add Splits File:**
- Include the [splits_final.pkl](https://github.com/HealthX-Lab/DeepSegAbdominal/blob/main/Preprocessing/splits_final.pkl) file into the directory where the dataset has been preprocessed.

### Train the models
To train the models, use the following command:
```bash
nnUNet_train 3d_fullres TRAINER_CLASS_NAME TASK_NAME_OR_ID 0
```
#### Supported Model Architectures and Corresponding TRAINER_CLASS_NAME

| Model Architecture                | TRAINER_CLASS_NAME                              |
|------------------------------------|-------------------------------------------------|
| 3DUNet model                       | nnUNetTrainerV2_noDeepSupervisionAndMirroring   |
| Four stage 3DUNet                  | nnUNetTrainerV2_noDeepSupervisionAndMirroringS4 |
| STUNet model                       | nnUNetTrainerV2_STUNet                          |
| Four stage STUNet model             | nnUNetTrainerV2_STUNetS5                        |
| AttentionUNet model                | nnUNetTrainerV2_AttentionUnet                   |
| Four stage AttentionUNet model      | nnUNetTrainerV2_AttentionUnetS4                 |
| SwinUNETR model                     | nnUNetTrainerV2_SUR                             |
| FocalSegNet model                   | nnUNetTrainerV2_FocalUNETR48                    |
| 3DSwinUnet model                    | nnUNetTrainerV2_SwinUnet                        |
| 3DSwinUnet with patch size 2 model  | nnUNetTrainerV2_SwinUnetP2                      |
| Four stage 3DSwinUnet with patch size 2 model | nnUNetTrainerV2_SwinUnetS4            |
| 3DSwinUnetV1 model                   | nnUNetTrainerV2_SwinUnetV1                      |
| 3DSwinUnetV2 model                   | nnUNetTrainerV2_SwinUnetV2                      |
| 3DSwinUnetV3 model                   | nnUNetTrainerV2_SwinUnetV3                      |
| 3DSwinUnetV4 model                   | nnUNetTrainerV2_SwinUnetV4   
| 3DSwinUnetB0 model                   | nnUNetTrainerV2_SwinUnetB0                      |
| 3DSwinUnetB1 model                   | nnUNetTrainerV2_SwinUnetB1                      |
| 3DSwinUnetB2 model                   | nnUNetTrainerV2_SwinUnetB2                      |
| 3DSwinUnetB3 model                   | nnUNetTrainerV2_SwinUnetB3                      |
### Inference
Follow these steps to perform inference on your desired input:

1. **Place Model Weights:**
   - Copy our model's weights into the directory specified for training models in the nnUNet framework (`nnUNet_trained_models`).

2. **Run Inference Command:**
   - Execute the following command to perform the inference:
     ```bash
     nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -tr TRAINER_CLASS_NAME --disable_tta
     ```
     Replace placeholders with actual values:
     - `INPUT_FOLDER`: Directory containing input data.
     - `OUTPUT_FOLDER`: Directory for storing the output predictions.
     - `TASK_NAME_OR_ID`: Task name or ID for the inference task.
     - `TRAINER_CLASS_NAME`: Trainer class name corresponding to the model architecture.
     
3. **Additional Information:**
   - For more detailed information on training and inference using the nnUNet framework, refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).
