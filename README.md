# Fellowship.ai Computer Vision Challenge

The problem statement requires the development of a DNN that is capable of classifying the 102 flower species present in the [Oxford Flowers102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). The DNN is based on a ResNet50 architecture pre-trained on the [ImageNet](https://www.image-net.org/) database.

## Dataset Description


## Approach

1. Preparing the Data: The data was split for train, test and validation in the manner of a train/test/dev set with the split being .75:.15:.15 respectively. The main motive behind splitting the data is to ensure that the model generalizes well on unseen data and not just on the trainining data.
2. Initializing the neural network: According to the problem statement, a ResNet50 model was initialized with weights from imagenet so that relevant features can be extracted from the train data.
3. 
