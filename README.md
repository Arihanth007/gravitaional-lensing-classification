# Gravitaional Lensing Classification

I implemented binary classification using Equivariant Convolutional Neural Networks and multi-class classification of gravitational lenses using Transfer Learning. The model classifies gravitational lensing images among ones with substructure, without substructure and vortex.

## Binary Classification - Equivariant CNNs

I achieve P4 equivariance on images which corresponds to equivariance in 90 degree rotations of the image. I achieved a test AUC score of 1.0 on a 80-20 split between train and test. I also achieve an AUC of 1.0 on rotated images belonging to the test set. I used the implementation of P4 group CNN from the paper 'Group Equivariant Convolutional Networks' - T. Cohen and M. Welling (2016).

## Multi-class Classification - CNNs

I applied transfer learning by using a pre-trained model - resnet18. I achived a test AUC score of 0.984 on a 75-25 split between train and test and a validation AUC of 0.958. The entire train and validation data wasn't fitting into the RAM and had to load the data in batches for training and testing.

|Model|Test AUC|
|-----|--------|
| Equi-CNN   | 1.0 |
| Multiclass-CNN | 0.984 |
