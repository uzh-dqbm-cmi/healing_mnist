# Convolutional Autoencoder for classical and healing MNIST

The goal of this research project is to explore different deep learning approaches to image analysis and if possible develop/implement a deep learning model being able to learn important features of sequences of images and make predictions/classifications. 

## Description of datasets

The dataset that will be used for training and test of the models is the well-known MNIST dataset which contains images of hand-written digits from 0-9. For simplicity, the images are kept black and white. This dataset is subsequently adapted for more advanced tasks following the example of the `Healing MNIST` implementation from ... From a single image of digit a sequence of digits is created by rotating the original image. As a further extension, there can be a square of specified or random dimensions added to the top left corner of the image at some point in the sequence and removed at a later point in the sequence. In a further extension, the square size can also be implemented dynamically, growing first and then shrinking in size. These modifications serve the purpose of simulating a series of medical data collections of a patient which are taken at specified intervals. The square in the corner may be representative of the presence of a very dominant feature (such as an illness, fever) which can be treated and become absent again. 

## Outline of project

Never having worked with neural networks before, the first tasks was to familiarize myself with deep learning models for image analysis. The most prominent of these models is the convolutional neural network. In a first step, I went over the most important aspects regarding architecture, training and evaluation of a convolutional neural network (CNN) for image classification. After successfully implementing a CNN for classification, I changed the task from image classification to image reconstruction. To achieve this goal, I implemented a convolutional autoencoder where the encoder part mapped the input images to a low-dimensional embedding (latent representation) via convolutional layers and the decoder part reconstructed the original image from the low-dimensional embedding via convolutionally transposed layers. One goal was to optimize the reconstruction of the images but in a further step I also explored how adding a second objective namely the classfication of the digits using only the low-dimensional embedding as input. It was analyzed how the perfomance of the classifier and the autoencoder as well as the visualization of the low-dimensional embedding (mapped onto 2D coordinates via t-SNE) changed with respect to different weights for the 2 objectives as well as the dimensionality of the low-dimensional embedding. 

The convolutional autoencoder was further used for reconstruction of images of the Healing MNIST dataset. In a first step, a square of constant size was added randomly to a specified proportion of the images. In further steps, the size of the square also varied randomly between images between a minimal and a maximal size. Later, rotations were applied to the dataset so that again a specified proportion of the images were rotated with a random angle between 0° and 180° degrees. The goal was to find an architecture  and training procedure of the convolutional autoencoder so that reconstruction as well as subsequent classification of the modified images from the low-dimensional embedding performed reasonably well. For this end, layer types, layer dimensions, different learning rates, different loss function as well as different regularization techniques were tested out to find the best performing model. 


In a last step, the goal was to move on from classification or reconstruction of single images to analysis of time series of images. The temporal component was implemented by starting with the original image and augmenting the sequence with sequential rotations of the original image. At some point in the sequence also the square in the top left corner could appear and also disappear again either with constant or dynamically growing and shrinking square size. The task at hand was to predict the next image from the input sequence. The method of choice was the implement a combination of a convolutional neural network to learn to the most important (spatially related) features from the images with a Long Short Term Memory network (LSTM) which captures the temporal aspect of the image time series. 


# Convolutional Neural Networks

## Components of CNNs

### Convolutional layer

### Pooling layer

### Activation layer

### Regularization layer

### Training and Evaluation 


## Implementation of CNN for MNIST classification 

and implemented a version for the MNIST dataset in Pytorch following the outline from the online tutorial by Nutan (https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118). 

The CNN follows a very simple architecture with 2 convolutional layers each followed by an activation layer with the ReLU activation function and a pooling layer. This simple architecture served as guide on how to implement a neural network, understanding the effect of the different layers as well as how to make the dimensions of the different layers compatible with each other. The theory behind the different layers of the convolutional network was nicely explained on the course website "Deep Learning for Computer Vision" from Stanford University and gave me the necessary understanding for trying out different architectures. As the MNIST dataset is already very well explored, the implemented architecture for classification achieved a high accuracy of 98% on a validation set. 

# Convolutional Autoencoder 

A classical autoencoder is an unsupervised deep learning model that learns to encode high-dimensional input data into a lower-dimensional latent representation, and then decode it back to the original space with minimal loss of information. The autoencoder consists of an encoder network that maps the input to the latent representation, and a decoder network that reconstructs the input from the latent representation. The goal of an autoencoder is to learn a compressed representation of the input data that captures its essential features in a lower-dimensional space. The low-dimensional embedding produced by the encoder is typically the most interesting aspect of the autoencoder, as it represents a compact and informative representation of the input data. This embedding can be used for a variety of downstream tasks, such as clustering, classification, or visualization. In our case, the low-dimensional embedding can be used as a tool to assess patient similarity by application of clustering algorithms or different assessment tools onto the low-dimensional embedding. 

In the case of a convolutional autoencoder, the input data is typically an image, and the encoder and decoder networks are composed of convolutional and pooling layers. The encoder network maps the input image to a lower-dimensional latent representation, while the decoder network reconstructs the input image from the latent representation. The encoder typically consists of a series of convolutional layers followed by pooling layers, which progressively reduce the spatial dimensions of the input image while increasing the number of channels. The output of the final convolutional layer is flattened and passed through one or more fully connected layers to produce the latent representation. The decoder network is the reverse of the encoder network, consisting of one or more fully connected layers followed by a series of transpose convolutional layers, which gradually increase the spatial dimensions of the latent representation while reducing the number of channels. The output of the final transpose convolutional layer is the reconstructed image. During training, the autoencoder is optimized to minimize the reconstruction loss between the input image and its reconstruction. The loss function typically involves a measure of pixel-wise difference between the input and reconstructed images, such as mean squared error or binary cross-entropy.

## Implementation of convolutional autoencoder from online tutorial

The architecture of the convolutional autoencoder was taken from the online tutorial by Eugenia Anello [Link](https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjVkZjFmOTQ1ZmY5MDZhZWFlZmE5M2MyNzY5OGRiNDA2ZDYwNmIwZTgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2NzgyNjQ3OTksImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExNTAxNDU2NjA3MjA3MTU5MDcxMCIsImVtYWlsIjoibWVuc21lbmdlckBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6Ik1lbGlzc2EgRW5zbWVuZ2VyIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FHTm15eGJ2UkFGMFNIeG4zWnI2Zms1YTRzcFd5LWxfbXJPS1duNXlkY2tZPXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6Ik1lbGlzc2EiLCJmYW1pbHlfbmFtZSI6IkVuc21lbmdlciIsImlhdCI6MTY3ODI2NTA5OSwiZXhwIjoxNjc4MjY4Njk5LCJqdGkiOiJiYThiOTBlZWI2YjU3MmU4Zjc3MjY0MmYyN2FjNWVhMzQ0NGY5Mjg0In0.2e_lGQi-miFFUXBuUvXH-2mw8neT3XeqzlkR66WC3DTOVAIx2yhqSDc2C3Z-zZNwSURQBU4ehiU7WbUmnqo0jGC9rlQL6d0VyCLl2pryQ0IijqdhiwIdoaHHz4xsB3Kps7piES1MIXYMHH01iSav_easASkXFSY4Q7RCj9SpYnuQ_fksg-ULOyawWf_eJB8pK2ApCx9jCQ-wxIOuzNgjbhlbv3Ds4Bwn0TlfoOAbx5GB2Tqbth9Hlhpy7RfHB8v--1-muIaAC8o6-l1TAReNEWUMts_ljiP205pYVAU_V-QkMf6YPqQBt4xjwUsxXikt2J1TnjoWoavPi3fL8eLh2A). 

The Encoder part of the convolutional autoencoder consists of 3 convolutional layers each followed by an activation layer (ReLU activation function). Instead of adding a pooling layer after each convolutional layer in order to downsize the image, the dimensionality was reduced by choosing a stride size of 2 and a kernel size of 3 for each of the convolutional layers. After the second convolutional layer, there was also a batch normalization layer added which should assist the faster convergence of the optimization problem. After the 3 convolutional layers with filters of dimension 8, 16 and 32 respectively, the last hidden representation was flattened to produce a linear layer. This served as input to 2 fully connected layers, the last of which having the dimensionality of the specified low-dimensional embedding. 

The Decoder part of the convolutional autoencoder mirrored the layers of the encoder network. The low-dimensional embedding served as input to 2 fully connected layers. The last hidden layer was "unflatted" to get again a 2-dimensional images representation as well as a third dimension representing the channels. Via 3 transpose convolutional layers with a stride of 2, the original dimensionality of the input image was achieved. In a last step the data was fed through a sigmoid layer to get normalized pixels between 0 and 1 again. 

During training the original input image and the reconstructed image as the output of the decoder network was compared via the mean squared loss which takes the mean squared error for each pixel between the original and the reconstructed image and averages over all the pixels. As the optimizer, the ADAM algorithm with a learning rate of 0.001 was chosen performing backpropagation over the weights of both the encoder and decoder simultaneously.

The implemented autoencoder performed very well on the original MNIST dataset with a mean squared error of only around 0.02. After training was completed, the low-dimensional embedding served as input to a very simple classifier MLP which achieved still a solid accuracy of 92% for a low-dimensional embedding of $d=5$. The classifier takes the low-dimensional embedding as input, maps it to a hidden layer of higher dimensions and then maps it again to the output layer with 10 nodes (one for each class). The class probability can be obtained by putting the output values through a sigmoid function. The loss function used for training was the crossentropy loss. 

The visualization of the low-dimensional embedding also shows clearly the separation for the different digit without this being learned. This indicates that the features extracted from the convolutional layers in the encoder which are important for reconstruction play also a big role for distinguishing between the different classes. 

%% TO DO:< insert image showing low-dimensional representation and reconstruction of d = 5>


## Simultaneous training of autoencoder and classifier

Instead of sequentially applying the autoencoder and classifier to the input images and separately training the 2 networks, the classifier is embedded into the training procedure of the autoencoder taking the current low-dimensional embedding from the encoder as input. For this, the 2 loss functions (MSE for the autoencoder and the crossentropy loss for the classifier) can simply be added together with a hyperparameter specifying the weight each loss function should have for the overall optimization. One caveat for this is that the losses operate on different scales. The crossentropy loss has the first non-zero digit at $10^0$ and the MSE at $10^{-2}$. Either there have to be different weights to be tested out or the the losses can be weighted equally by weighing each loss by term consisting of the current value of the other loss divided by the sum of the current values of both losses. 

For this simultaneous training, the trade-off in the accuracy of the classifier vs. the reconstruction error of the autoencoder as well as the changes in the low-dimensional embedding are evaluated for different dimensionalities of the low-dimensional embedding and different weights for the 2 losses. 


### Impact of hyperparameter dimensionality

For evaluating the trade-off between accuracy of the classifier and the reconstruction error for different dimensionalities of the low-dimensional embedding, the losses were weighted equally following the method outlined before. The low-dimensional embedding was varied from $d=3$ up to $d=10$. As expected the reconstruction error decreased with higher dimensionality of the embedding, since more information could be retained. However, the accuracy of the classification had its optimum at $d=5$ indicating that there is a potential for overfitting. 


### Impact of hyperparameter for weighting losses

For evaluating the trade-off between accuracy of the classifier and the reconstruction error for different weights of the respective losses, the dimensionality of the embedding was kept fixed at $d=5$ since this showed a good performance for both the classifier and the autoencoder. The overall loss was calculated according to the following loss function: 
$$
loss =  \lambda * loss_{autoencoder} + (1-\lambda)* loss_{classifier}
$$
where $\lambda = 0$ means that the parameters of the network were only optimized with respect to the performance of the classifier and for $\lambda = 1$ the parameters were only optimized with respect to the performance of the reconstruction. The overall performance (lowest overall loss) was achieved for ... which comes close to weighting the losses equally where the weight parameter after the last iteration was recorded at ... 


## Modification of convolutional autoencoder to improve performance and robustness

As mentioned in the project outline, the images were modified by adding a small square (initially 5x5 pixels) into the top left corner to a certain percentage of the images. The performance of the previously implemented classifier with its original architecture and training procedure was evaluated. Unfortunately, the reconstruction of the images of the dataset with the square performed very badly and the training as well as test loss remained very high after 30+ training epochs. This may be attributed to the fact that the MSE square loss is very sensitive to noise and large deviations of the reconstructed pixel values to the original pixel values. 

It therefore became necessary to either change the hyperparameter of the training procedure such as the loss function, the learning rate or the choice of optimizer or the architecture of the autoencoder itself by adding or removing different layers, addition of regularization layers and/or changing the activation function. Some of those options were tried out, the performance on the test dataset recorded and from there the best option regarding model architecture and training procedure was chosen for further analysis. 

### Loss function

The first obvious hyperparameter to vary is the choice of the loss function. As mentioned before, the MSE loss is very sensitive towards deviations between original and reconstructed image since large losses of single pixels are weighted disproportionally large. A first alternative would be the L1-loss which takes the absolute difference (and not the squared) difference between original and estimated pixel value and therefore does not over-penalize large errors. Changing the loss function from the L2-loss to the L1-loss improved the reconstruction quality significantly. The square in the corner was also picked up and reconstructed correctly. 

Other loss functions such as the SSIM-loss (structural similarity loss) and the binary entropy loss were also tried out but performed equally bad or worse than the L2-loss. As the L1-loss overwhelmingly improved model performance, the rest of the assessment was conducted with the L1-loss as the chosen loss function. 

### Activation function

The ReLU activation function is prone to undesirable effects such as "dying ReLU" or vanishing gradients. To overcome these drawbacks other, still very similar activation functions such as `Leaky ReLU` or `ELU` can be used to replace the ReLU activation function. However, this had no significant effect on model performance. 

### Architecture & additional layers of encoder/decoder

Adding more convolutional of fully connected layers to both the encoder and the decoder part of the autoencoder as well as adding more filters to the convolutional layers of increasing the size of the hidden fully connected layers could improve model performance since the model can learn more features and has greater flexibility. However, it can also lead to overfitting as well as increasing the chance of getting stuck at a local optimum in the optimization process. Moreover, a more complex model needs more parameters to be optimized and therefore increases the computational effort drastically. 

More convolutional and/or fully connected layers were added to the model but the loss on the test dataset after training was increased compared to the original architecture. Similar outcomes were observed when either increasing the number of filters for the convolutional filters of both the encoder or decoder or increasing the number of nodes in the linear layers for both the encoder and the decoder. 

Another approach was to downsize the image between the convolutional layers not via the stride parameter in the convolutional layers but by adding a pooling layer between the convolutional layers and setting the stride to 1 for the convolutional layers. 

### Regularization layers
Since increasing the complexity of the model did not improve the model performance, one approach could be to introduce more regularization layers such as batch normalization layers and dropout layers. Applying dropout layers with dropout probability 


## Performance of 2 selected improved convolutional autoencoders

### Classification accuracy

### Varying square ratio

### Varying square size

### Multiple squares?





