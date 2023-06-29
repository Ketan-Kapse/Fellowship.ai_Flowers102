# Fellowship.ai Computer Vision Challenge

The problem statement requires the development of a DNN that is capable of classifying the 102 flower species present in the [Oxford Flowers102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). The DNN is based on a ResNet50 architecture pre-trained on the [ImageNet](https://www.image-net.org/) database.

## Dataset Observations

1. Large Number of Classes: The flowers102 dataset consists of a large number of flower species such  as roses, orchids, sunflowers, etc. Each class represents a distinct species, and exhibit variations in color and shape (the species can come in different colors like reds, blues and pinks, and their petals can range from being round to elongated) allowing for indepth classification.

    ![examples](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/fd703a73-c30f-499d-b2b5-1ca9700a23d8)


2. Potential Classification Issues: The classes in this dataset are not balanced, i.e. some classes have more number of images associated with them than others. This imbalance can lead to the model being biased towards the classes with more images which is indesirable. Furthermore, there can be the presence of visual similarites between different varieties of flowers which can make it difficult to distinguish between them. Also, the backgrounds of the different images can vary which adds to the classification complexity.

    ![classDist](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/f11f0c0c-3998-4a3e-a99f-ecbf30da0e04)


3. Different Image Sizes: The images in the dataset are of different dimensions, with some images being larger than the others. This requires resizing of all the images in the dataset to a uniform dimension before they are fed to the network.

    ![height](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/caf76300-631b-4c66-98ee-2d69fb0ce321)![width](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/434113d7-4f18-424e-98f8-826f66d349ba)




## Approach

1. Preparing the Data: The data was split for train, test and validation in the manner of a train/test/dev set with the split being .75:.15:.15 respectively. The main motive behind splitting the data is to ensure that the model generalizes well on unseen data and not just on the trainining data.
2. Initializing the neural network: According to the problem statement, a ResNet50 model was initialized with weights from imagenet so that relevant features can be extracted from the train data.

## Model Architecture and Training
For this dataset, the ResNet50V2 architecture pretrained on the imagenet database is used. Custom fully connected and dropout layers have been added on top of the resnet architecture to improve network complexity. Furthermore, the resnet layers are frozen so as to avoid retraining them again. 

![model_plot](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/45aa165d-8bf4-4ec8-8345-17267c5a44a2)

The final model contains approx. 24.77M parameters of which 1.2M are trainable.

![model_summary](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/69478121-c0f9-4670-b985-811b9dd42011)


The input training images are subject to transformations such as random rotations, flips, etc. before they are fed to the network in batches of size 32. These augmentations improve the diversity of the dataset, reducing overfitting and improving generalization on the test set.

```
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.4),
  tf.keras.layers.Resizing(224, 224),
  tf.keras.layers.Rescaling(1./255)
])

```

Next, the model was trained for 45 epochs with callbacks like ReduceLROnPlateau and ModelCheckpoints being implemented.

## Results
This model demonstrates an accuracy of ~91% on the training set and ~88% on the validation set.

![accuracy](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/efe4821e-be3c-4fe7-8663-c1abd2664366)![loss](https://github.com/Ketan-Kapse/Felloship.ai_Flowers102/assets/47895059/7e3a2875-74c1-4df0-8e37-82339388f9e0)



## Further Improvements
While this model demonstrates satisfactory results, they can be further improved with the following considerations:
1. Increasing the size of the data: The flowers 102 dataset is small, and further expansion of the dataset with the addition of more images can help improve performance.
2. Fine Tuning and Using Different Models: Further study of the hyperparameter tuning like LR and regularization can help improve the model's performance. Additionally, experimentation with different architectures such as VGGs or MobileNets can lead to the discovery of better performance. Also, the usage of more complex ResNet architectures like ResNet101 or ResNet 152 can lead to better feature recognition, increasing accuracy.
