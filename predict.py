from modelparameter import ModelParameters,constructModel
from preprocessing import process_image,imshow
from torch.cuda import is_available
from torch import optim,nn,device,no_grad,exp,mean,FloatTensor,save,load,from_numpy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg13_bn,alexnet
from collections import OrderedDict
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from modelparameter import ModelParameters,constructModel,load_checkpoint
import argparse
import os
import glob
from utils import getLabelDict
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img=Image.open(image_path).convert('RGB')
    torchimg=from_numpy(process_image(img))
    model.eval()
    if (model.gpu_request == True):
        selected_device = device("cuda" if is_available() else "cpu")
    else:
        selected_device = "cpu"
    with  no_grad():
        torchimg=torchimg.unsqueeze(0)
        inputs=torchimg.to(selected_device)
        inputs=inputs.float()
        logps=model(inputs)
        ps = exp(logps)
        (top_p, top_class) = ps.topk(topk)
    return top_p.squeeze().tolist(),top_class.squeeze().tolist()


def display(filename,model,probs,classes):
    '''
        Display an image along with the top x classes
    '''
    
    labels = []
    for c in classes:
        labels.append(model.cat_to_name[model.idx_to_labels[c]])
    label_pos = np.arange(len(labels))
    #truth_label = model.cat_to_name[truth]
    img = Image.open(filename).convert('RGB')
    torchimg = from_numpy(process_image(img))
    '''
    display the image if it is not a console application
    '''
    print("\nThe flower image was classified as : "+labels[0])
    print("\nTop classifications are : "+str(labels))
    print("\nTop probabilities are : "+str(probs))
    


def main():
    parser = argparse.ArgumentParser(description='Inference Application')
    parser.add_argument('img_file', type=str, help='Filepath of image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint name, enter the model name such as vgg16_bn, vgg13_bn or alexnet')
    parser.add_argument('-t','--top_k', type=int, help='Number of top classes to be displayed. default is 3',default = 3)
    parser.add_argument('-c','--category_names', type=str, help='Path to the json file with category names. default path is root/cat_to_name.json',default = "cat_to_name.json")
    parser.add_argument('-p', '--gpu',help='Add this flag to enable gpu if available',action='store_true')
    args = parser.parse_args()    
    params = ModelParameters('\0')
    if os.path.exists("model_params"): 
        model_params = open("model_params", mode="rb")
        params = pickle.load(model_params)
        model_params.close()
        params.arch = args.checkpoint
    else:
        print("Cannot locate save directory of the checkpoint")
        print("\npossible solution: Run the train.py first")
        return 0
    cat_to_name = getLabelDict(args.category_names)
    if(cat_to_name is not None):
        save_path= params.saveDir+'/'+args.checkpoint
        model = load_checkpoint(save_path,params)
        if(model is not None):
            model.gpu_request = args.gpu
            probs,classes = predict(args.img_file,model,args.top_k)
            model.cat_to_name = cat_to_name
            display(args.img_file,model,probs,classes)
            
if __name__ == "__main__":
    main()