from torch.utils.data import DataLoader
import pickle
from torch.nn import functional
from torch.cuda import is_available
from torch import optim,nn,device,no_grad,exp,mean,FloatTensor,save,load,from_numpy
from collections import OrderedDict
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg13_bn,alexnet,vgg16_bn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from modelparameter import ModelParameters,constructModel,load_checkpoint
import argparse
import os
import glob
__version__ = "1.0.0"


def dispVersion():
    print(__version__)


def loadData(data_dir,batch_size=64):
    train_dir = data_dir+"/train"
    valid_dir = data_dir+"/valid"
    test_dir = data_dir+"/test"
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {'val_test_transforms': transforms.Compose([transforms.Resize(255),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225))]),
                       'train_transforms': transforms.Compose([transforms.RandomHorizontalFlip(),
                                                               transforms.RandomRotation(degrees=30),
                                                               # -30<=x<=30 degrees of random transform
                                                               transforms.RandomResizedCrop(224),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                                    (0.229, 0.224, 0.225))])}
    # Load the datasets with ImageFolder
    image_datasets = {'train': ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                      'val': ImageFolder(valid_dir, transform=data_transforms['val_test_transforms']),
                      'test': ImageFolder(test_dir, transform=data_transforms['val_test_transforms'])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataLoaders = {'train_loader': DataLoader(dataset=image_datasets['train'], batch_size=batch_size, shuffle=True),
                   'val_loader': DataLoader(dataset=image_datasets['val'], batch_size=batch_size, shuffle=True),
                   'test_loader': DataLoader(dataset=image_datasets['test'], batch_size=batch_size, shuffle=True)}
    return image_datasets,dataLoaders


def trainingPipeline(model,params,dataloaders):
    # Training the model
    if(params.gpu):
        selected_device = device("cuda" if is_available() else "cpu")
    else:
        selected_device = "cpu"
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=params.learnRate)
    model.to(selected_device)
    epochs = params.epochs
    steps = 0
    print_every = 10
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train_loader']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(selected_device), labels.to(selected_device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with  no_grad():
                    for valInputs, valLabels in dataloaders['val_loader']:
                        valInputs, valLabels = valInputs.to(selected_device), valLabels.to(selected_device)
                        logps = model.forward(valInputs)
                        batch_loss = criterion(logps, valLabels)
                        val_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == valLabels.view(*top_class.shape)
                        accuracy += mean(equals.type(FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Val loss: {val_loss / len(dataloaders['val_loader']):.3f}.. "
                      f"Val accuracy: {accuracy / len(dataloaders['val_loader']):.3f}")
                running_loss = 0
                model.train()
    model.optimizer = optimizer
    return model


def testModel(model,params,testloader):
    # Do validation on the test set
    if (params.gpu):
        selected_device = device("cuda" if is_available() else "cpu")
    else:
        selected_device = "cpu"
    test_loss = 0
    accuracy = 0
    model.eval()
    criterion = nn.NLLLoss()
    with  no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(selected_device), labels.to(selected_device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += mean(equals.type(FloatTensor)).item()

        print(f"Test loss: {test_loss / len(testloader):.3f}.. "
              f"Test accuracy: {accuracy / len(testloader):.3f}")
    return top_p, top_class

def saveCheckpoint(model,image_datasets,params):
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_path = params.saveDir+'/'+params.arch
    model.N_epochs = params.epochs
    # Save the model state dictionary as opposed to the whole model
    if os.path.exists(params.saveDir):
        if os.path.exists(save_path):  
            files = glob.glob(save_path+"/*")
            for f in files:
                os.remove(f)
        else:
            os.mkdir(save_path)
    else:
        os.mkdir(params.saveDir)
        os.mkdir(save_path)
    model_state_dict = save_path+"/model_state_dict"
    others = save_path+"/other_state_dicts"
    model_truths =  save_path+"/model_truths"
    model_params = open("model_params", mode="wb")
    save(model.state_dict(), model_state_dict)
    save({"optimizer_state": model.optimizer.state_dict(), "Epochs": model.N_epochs}, others)
    save(model.class_to_idx, model_truths)
    pickle.dump(params, model_params)

def main():
    parser = argparse.ArgumentParser(description='training application')
    parser.add_argument('data_dir', type=str, help='data dir for images')
    parser.add_argument('-s','--save', type=str, help='Save dir for checkpoint',default = "model_save")
    parser.add_argument('-a','--arch', type=str, help='architecture to be used : default is vgg16_bn. other models that can be chosen are alexnet and vgg13_bn ',default = 'vgg16_bn')
    parser.add_argument('-r','--learning_rate', type=float, help='learning rate for the model. default is 0.003',default = 0.003)
    parser.add_argument('-d','--hidden_units', type=int, help='number of hidden units, default is 4096',default = 4096)
    parser.add_argument('-e','--epochs', type=int, help='epochs for the model. default is 20',default=20)
    parser.add_argument('-p', '--gpu',help='add this flag to enable gpu if available',action='store_true')
    args = parser.parse_args()
    params = ModelParameters(dataDir=args.data_dir)
    if os.path.exists("model_params"): 
        model_params = open("model_params", mode="rb")
        params = pickle.load(model_params)
        model_params.close()
    params.dataDir = args.data_dir
    params.addArch(args.arch)
    params.updateSaveDir(args.save)
    params.setGpuPreference(args.gpu)
    params.setHidden(args.hidden_units)
    params.setLearnRate(args.learning_rate) 
    params.setEpochs(args.epochs)
    image_datasets , dataloaders = loadData(params.dataDir)
    model = constructModel(params)
    print(model)
    model = trainingPipeline(model,params,dataloaders)
    saveCheckpoint(model,image_datasets,params)
    testModel(model,params,dataloaders['test_loader'])
    
if __name__ == "__main__":
    main()