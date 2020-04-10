# Mnist-Conv-Keras
Start here if...

You have some experience with R or Python and machine learning basics, but you’re new to computer vision. This competition is the perfect introduction to techniques like neural networks using a classic dataset including pre-extracted features.
Competition Description

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.
Practice Skills

    Computer vision fundamentals including simple neural networks

    Classification methods such as SVM and K-nearest neighbors

Acknowledgements 

More details about the dataset, including algorithms that have been tried on it and their levels of success, can be found at http://yann.lecun.com/exdb/mnist/index.html. The dataset is made available under a Creative Commons Attribution-Share Alike 3.0 license.

This Project Consists of 50000+ images of Digits which is then Categorized by our Neural Network 
We Also Learn about a new layer which is called Convolution Layer

# Convolution Layer
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

# Neural Network
A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.[1] Thus a neural network is either a biological neural network, made up of real biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modeled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.

![Image of a Convoluted Neural Network](https://adeshpande3.github.io/assets/Cover.png)


# Standardize
A standard approach is to scale the inputs to have mean 0 and a variance of 1. Also linear decorrelation/whitening/pca helps a lot.

# Dense Layer

A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a. The following is te docstring of class Dense from the keras documentation:
output = activation(dot(input, kernel) + bias)where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.

# Dropout Layer 

Dropout is a a technique used to tackle Overfitting . The Dropout method in keras.layers module takes in a float between 0 and 1, which is the fraction of the neurons to drop. Below is the docstring of the Dropout method from the documentation:
Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

# Compile

Every Neural Network should be compiled before Training it on a Dataset. During Compilation wer have to provide our neural network with an optimizer, a loss function as well as the Metrics that we need to observe during Training

## Adam
Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

## Categorical Cross-Entropy
One-of-many classification. Each sample can belong to ONE of C classes. The CNN will have C output neurons that can be gathered in a vector s (Scores). The target (ground truth) vector t will be a one-hot vector with a positive class and C−1 negative classes.
This task is treated as a single classification problem of samples in one of C classes.

## Softmax
Softmax it’s a function, not a loss. It squashes a vector in the range (0, 1) and all the resulting elements add up to 1. It is applied to the output scores s. As elements represent a class, they can be interpreted as class probabilities.
The Softmax function cannot be applied independently to each si, since it depends on all elements of s. For a given class si, the Softmax function can be computed as:
![Softmax](https://latex.codecogs.com/gif.latex?f(s)_{i}&space;=&space;\frac{e^{s_{i}}}{\sum_{j}^{C}&space;e^{s_{j}}})

# Mnist dataset have been along for a long time and a single Convolution Layer with some dense layer can help achieve very high accuracy. Some of the plots from running this file is given below 

![Accuracy](https://raw.githubusercontent.com/Mkaif-Agb/Mnist-Conv-Keras/master/Acc.png)
![Loss](https://raw.githubusercontent.com/Mkaif-Agb/Mnist-Conv-Keras/master/Loss.png)
