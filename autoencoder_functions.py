import numpy as np
import torch 
import matplotlib.pyplot as plt
import sklearn
import torchvision
from torchvision import datasets
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import pandas as pd
import random
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm




###### Functions for autoencoder


## Construction of encoder part
class Encoder_original(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            #nn.Dropout(p=0.2),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #nn.Dropout(p=0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            #nn.Dropout(p=0.2)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

## Construction of decoder part
class Decoder_original(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
    

def train_square(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


## Evaluation of model on test dataset
def evaluate_square(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
    val_loss = loss_fn(conc_out, conc_label)
    return val_loss



############# Functions for plotting

def plot_ae_outputs(test_dataset,device,encoder,decoder,n=10):
    fig = plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = fig.add_subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = fig.add_subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    fig.show() 
    return fig


def plot_ae_outputs_square(test_dataset,device,encoder,decoder,n=10):
    fig = plt.figure(figsize=(16,4.5))
    targets = []
    for _, target, _ in test_dataset:
        targets.append(target)
    targets = np.array(targets)
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = fig.add_subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = fig.add_subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    fig.show() 
    return fig
    
def plot_ae_outputs_square_custom(test_dataset, device, encoder, decoder, n=10, class_num=None):
    fig = plt.figure(figsize=(16, 4.5))
    targets = []
    for _, target, _ in test_dataset:
        targets.append(target)
    targets = np.array(targets)
    if class_num is not None:
        idxs = np.where(targets == class_num)[0][:n]
    else:
        idxs = []
        for i in range(len(np.unique(targets))):
            idx = np.where(targets == i)[0]
            idxs.append(np.random.choice(idx))
        idxs = np.array(idxs)
    t_idx = {i: idxs[i] for i in range(n)}
    for i in range(n):
        ax = fig.add_subplot(2, n, i+1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original images')
        ax = fig.add_subplot(2, n, i+1+n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Reconstructed images')
    fig.show()
    #return fig


########### Functions for embeddings

def embedding(data, device, encoder):
    encoded_samples = []
    for sample in tqdm(data):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img  = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    return encoded_samples


def embedding_with_square(data, device, encoder):
    encoded_samples = []
    for sample in tqdm(data):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        square = sample[2]
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img  = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_sample['square'] = square
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    return encoded_samples


#### Functions for downstream classifier

class MLP_Classifier(nn.Module):
    def __init__(self, low_d):
        super(MLP_Classifier, self).__init__()
    
        ## forward layers
        self.layers = nn.Sequential(
            nn.Linear(low_d, out_features=25),
            nn.ReLU(),

        )
        self.out = nn.Sequential(
            nn.Linear(25,10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)
        return x

def train_classifier(loss_fn, model, dataloader, optimizer, n):
    for epoch in range(n):
        model.train()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            labels = labels.flatten()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{n}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}' )

            

def evaluate_classifier(model, dataloader):
    #model.evaluate()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            labels = labels.flatten()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    return (100 * correct / total)


def evaluate_classifier_classwise(model, dataloader):
    with torch.no_grad():
        num_classes = 10
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        for inputs, labels in dataloader:
            labels = labels.flatten()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(num_classes):
                class_labels = (labels == i)
                class_predicted = (predicted == i)
                class_correct[i] += (class_predicted & class_labels).sum().item()
                class_total[i] += class_labels.sum().item()

        for i in range(num_classes):
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of class {i}: {accuracy:.2f}%")
        
        overall_accuracy = 100 * sum(class_correct) / sum(class_total)
        print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")


############## Functions for simultaneous training

## definition of training function 

def co_train(loss_p, dataloader, encoder, decoder, classifier, loss_fn_classifier, loss_fn_autoencoder, optimizer, device):
    encoder.train()
    decoder.train()
    classifier.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, label_batch in dataloader: 
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        
        #use data additionally as input into classifier
        classified_data = classifier(encoded_data)

        #joint loss function 
        loss_autoencoder = loss_fn_autoencoder(decoded_data, image_batch)
        loss_classifier = loss_fn_classifier(classified_data, label_batch)
        if (loss_p == "inverse"):
            w1 = (loss_classifier/(loss_autoencoder+loss_classifier))
            w2 = (loss_autoencoder/(loss_autoencoder+loss_classifier))
            loss  = (loss_classifier/(loss_autoencoder+loss_classifier))*loss_autoencoder + (loss_autoencoder/(loss_autoencoder+loss_classifier))*loss_classifier
        else:
            w1 = loss_p
            w2 = 1-loss_p
            loss = loss_p*loss_autoencoder + (1-loss_p)*loss_classifier
            
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss_autoencoder.data))
        train_loss.append(loss.detach().cpu().numpy())
        

    return np.mean(train_loss), w1, w2 

## definition of evaluation function 

def co_evaluate(encoder, decoder, classifier, device, dataloader, loss_fn_classifier, loss_fn_reconstruction, loss_p):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        class_out = []
        class_label = []
        for image_batch, label_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Classify encoded data
            classified_data = classifier(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
            class_out.append(classified_data)
            class_label.append(label_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        class_out = torch.cat(class_out)
        class_label = torch.cat(class_label)
        # Evaluate global loss
    reconstruction_loss = loss_fn_reconstruction(conc_out, conc_label)
    classification_loss = loss_fn_classifier(class_out, class_label)
    if (loss_p == "inverse"):
        val_loss = (classification_loss/(classification_loss+reconstruction_loss))*reconstruction_loss + (reconstruction_loss/(classification_loss+reconstruction_loss))*classification_loss
    else:
        val_loss = loss_p*reconstruction_loss + (1-loss_p)*classification_loss
    return reconstruction_loss, classification_loss, val_loss


def co_evaluate_classifier(model, dataloader, encoder):
    #model.evaluate()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            #labels = labels.flatten()
            outputs = model(encoder(inputs))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')