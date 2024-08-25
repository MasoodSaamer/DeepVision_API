# main.py handles the training and learning part for the application
import torch
import torch.utils
import torch.utils.data
import torchvision 
from torchvision import transforms #to use the transform function
from torchvision import datasets #importing CIFAR-10 datasets, also includes datasets like ImageNet and more
from torchvision.models import resnet18, ResNet18_Weights

# To speed up the process, use GPU if its available or else just use CPU
# Note that using CPU will be significantly slower
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


#Step 1 to define the image transformation
#first we prepare the pre-processing images from CIFAR to ensure they can correctly be fed into the neural  network
#CIFAR-10 is 32x32 but since my model uses ResNEt, I have to resize the images to 224x224
#we can use the transform.Compose([]) function for this since it allows multiple image transformations in a sequential order
transform = transforms.Compose([transforms.Resize((224, 224)),  #resizing the image from 32x32 to 224x224 since we are using ResNet
                                 transforms.ToTensor(), #converting the image from PIL(or NumPy array) to tensor since PyTorch works with tensors
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalizing is basically used to convert the pixel values of RGB channels into something that can be read by the neural network. In our case we choose the mean for each RGB channel as (0.5, 0.5, 0.5) with the same 3d deviation which scales pixel values from [0, 1] to  range of [-1, 1]
                                 ])

#Step 2. Loading the CIFAR-10 datasets and applying transformation to them
#initializing trainset for CIFAR-10 using a trainset object which is an instance of the CIFAR-10 datasets
# Use a small subset of CIFAR-10 for quick testing
trainset = datasets.CIFAR10(root='./data/cifar10', #specifying the root directory to download CIFAR datasets
                            train=True,  #cifar has two datasets: training and test. this is to ensure it imports train set
                            download=True, #automatically downloads the files if not already downloaded
                            transform=transform) #applying the transformation from earlier
#now to create a trainloader which is a data loader for the training set. essentially, it is an iterator that loads the trainset data in batches. in our case, we get a batch of 32 images shuffled and preprocessed for each time we iterate through trainloader.
trainloader = torch.utils.data.DataLoader(trainset, #the dataset 
                                           batch_size=32, #specify the number of images to be loaded per each batch during training. in our case, during each iteration, 32 images will be processed at once
                                           shuffle=True, #shuffling the dataset of each epoch(one complete pass through the training data) to ensure the models dont learn the order of data. they are suppoesed to learn the actual features of the image instead
                                           num_workers=4) #how many subprocesses we should allocate. more workers can speed up but uses more memory
#now we will be creating our testsets starting with test images from cifar
testset = datasets.CIFAR10(root='./data/cifar10',
                           train=False, #ensuring we use test sets this time
                           download=True,
                           transform=transform)
#data loader for test set
testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=32, 
                                          shuffle=False, #unlike the trainset, we dont need to shuffle for the test set since its a fixed set.
                                          num_workers=4)


#This step is just to ensure the program works fine till now
#print(f'Total training samples: {len(trainset)}')
#print(f'Total testing samples: {len(testset)}')

#now we import a model that has already been trained on
#Step 3 implementing transfer learning 
#first we import some pre-trained model
from torchvision import models #provides access to several popular neural network architectures that have been pre-trained on large datasets like ImageNet

#I will be using ResNet18. the weights have already been trained on
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 

from torch import nn #nn contains essential neural network components such as layers, loss functions and activation functions

#now to replace the final layers
num_ftrs = model.fc.in_features #to get the number of input features to the fully connected(fc) layer of ResNet18 model
#simply put, the ResNet18 model has a fully connected layer called fc at the end which is responsible for producing the final classification outputs. so we extract the number of input features that this fc layers expects
#its needed as to ensure the new layer we are creating knows how many input features to expect.

#now to convert the layer from ResNet18 model to CIFAR model
model.fc = nn.Linear(num_ftrs, 10) #creates a new fc layer with num_ftrs as the input feature but limits it to 10 output features as cifar only has 10 classes 
model = model.to(device)

#Step 4. Now that we have transfered the ResNet18 model to our CIFAR case, lets work on training the model
from torch import optim #optim contrains various optimization algorithms used to adjust the weights of the neural network. Helps minimize the loss during training

criterion = nn.CrossEntropyLoss() #a loss function which helps evalute the performance of the model's predictions to how well they match the actual labels. It combines LogSoftmax and NLLLoss. 
optimizer = optim.Adam(model.parameters(), lr=0.001) #an optimizer which updates the model's weight gradients during training. Has a learning rate of 0.001. Uses the Adam optimizer.

# Guard to handle multiprocessing issues on Windows. guard ensures that the code inside it is only executed when the script is run directly, preventing multiprocessing issues on Windows.
if __name__ == '__main__': 

    #now we have to run a loop to see how the training model performs. Lets use 10 epochs
    #for each loop, it performs forward pass, calculates loss and then performs a backwards pass to update the weights. After each epoch, it shows the average loss for that epoch
    for epoch in range(10):
        running_loss = 0.0 #running loss is 0 at start. changes as the epoch executes
        for inputs, labels in trainloader: #an inner loop to iterate through the trainloader. it uses the input images and their corresponding labels. works one batch at a time. standard practice in Deep learning
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() #ensures the gradients of the model parameters are reset so they dont conflict with the next iteration.
            outputs = model(inputs) #forward pass. basically it passes the current batch of the input image through the model and generates the prediction as 'outputs'
            loss = criterion(outputs, labels) #Uses the CrossEntropyLoss function to compare the model's predictions(outputs) to the labels. returns the value quantified. More minimized, better
            loss.backward() #Backward pass used to compute the gradient of the loss with respect to the model's parameters. Updates the model's weight to be in the direction that minimized loss. also known as backpropagation
            optimizer.step() #based on the loss, the Adam optimizer uses that data to update the model's weight to be in the direction that minimized loss
            running_loss += loss.item() #adding all the loss values to the running_loss. the loss.item() function converts loss into a python number
            print(f'{running_loss}')
        print(f'Epoch {epoch + 1}, Loss: {running_loss/len(trainloader)}') #displaying the current epoch and the average loss for that epoch. Decreasing loss is what we want
    print(f'Finished training')

    #Step 5. Now to validate the model
    correct = 0 #count of how many correct predicitons the model made
    total = 0 #total number of predictions
    with torch.no_grad(): #this ensures the gradient calculation is disabled since we dont need to compute gradients for validation. Saves memory and computation time
        for inputs, labels in testloader: #Now we iterate over the testloader instead to see how our trainloader model compares
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) #forward pass to get the output based on giving the model the input image
            _, predicted = torch.max(outputs.data, 1) #Selects the class with the highest score (probability) as the predicted class within the output
            total += labels.size(0) #add the total number of images by batch size
            correct += (predicted == labels).sum().item() #adds the number of times the prediction matches the actual label in that batch
    print(f'Accuracy on the test images: {100 * correct/total}%') # % of how many correct predictions we got

    torch.save(model.state_dict(), './models/cifar10_resnet18.pth') #saving the trained model's learned parameters(state dictionary) so that it can be later used to make predictions, continue training or deploy to an application
    
