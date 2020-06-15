from torchvision.models import vgg13_bn,vgg16_bn,alexnet
from torch.nn import functional
from torch.cuda import is_available
from torch import optim,nn,device,no_grad,exp,mean,FloatTensor,save,load,from_numpy
from collections import OrderedDict
import os
class ModelParameters:
    '''
    default model is chosen as vgg13_bn. 
    The user can choose one additional model. 
    '''
    def __init__(self,dataDir):
        self.dataDir = dataDir
        self.saveDir = '\0'
        self.learnRate = 0.003
        self.epochs = 6
        self.nHiddenUnits = {'vgg16_bn' : 4096}
        self.arch = 'vgg16_bn'
        self.gpu =False

    def updateSaveDir(self,pathname):
        self.saveDir=pathname

    def addArch(self,Arch):
        self.arch = Arch
        
    def setGpuPreference(self,gpu):
        self.gpu = gpu
    
    def setHidden(self,nHidden):
        self.nHiddenUnits[self.arch]=nHidden
    
    def setEpochs(self,epochs):
        self.epochs = epochs
        
    def setLearnRate(self,learnRate):
        self.learnRate = learnRate
        
def constructModel(params):
    ## Build the network
    # Load a pretrained network
    # VGG13 was used with batch normalization as a starting point.
    # if the model performs well VGG 11 would be tried to reduce layers and vice-versa.
    model=None
    if params.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=True)
        # fc1 layer is unique for each NN
        fc1 = ('fc1', nn.Linear(25088, params.nHiddenUnits[params.arch]))
        fc2 = ('fc2', nn.Linear(params.nHiddenUnits[params.arch], 256))
    elif(params.arch == 'vgg13_bn'):
        model = vgg13_bn(pretrained=True)
        # fc1 layer is unique for each NN
        fc1 = ('fc1', nn.Linear(25088, params.nHiddenUnits[params.arch]))
        fc2 = ('fc2', nn.Linear(params.nHiddenUnits[params.arch], 256))
    elif(params.arch == 'alexnet'):
        model = alexnet(pretrained=True)
        # fc1 layer is unique for each NN
        fc1 = ('fc1', nn.Linear(9216, params.nHiddenUnits[params.arch]))
        fc2 = ('fc2', nn.Linear(params.nHiddenUnits[params.arch], 256))
    else:
        return None
    # freezing params
    for param in model.parameters():
        param.requires_grad = False
    # Untrained dense layers        
    classifier = nn.Sequential(OrderedDict([
            ('drop1',nn.Dropout(p=0.2)),
            fc1,
            ('relu', nn.ReLU()),
            ('drop2',nn.Dropout(p=0.2)),
            fc2,
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    model.classifier = classifier
    return model

def load_checkpoint (save_path,params):
    '''
    returns the model based on the stored files
    '''
    model=None
    if("vgg16_bn" in save_path):
        model = vgg16_bn(pretrained=True)
        # fc1 layer is unique for each NN
        fc1 = ('fc1', nn.Linear(25088, params.nHiddenUnits[params.arch]))
        fc2 = ('fc2', nn.Linear(params.nHiddenUnits[params.arch], 256))
    elif("vgg13_bn" in save_path):
        model = vgg13_bn(pretrained=True)
        # fc1 layer is unique for each NN
        fc1 = ('fc1', nn.Linear(25088, params.nHiddenUnits[params.arch]))
        fc2 = ('fc2', nn.Linear(params.nHiddenUnits[params.arch], 256))
    elif("alexnet" in save_path):
        model = alexnet(pretrained=True)
        # fc1 layer is unique for each NN
        fc1 = ('fc1', nn.Linear(9216, params.nHiddenUnits[params.arch]))
        fc2 = ('fc2', nn.Linear(params.nHiddenUnits[params.arch], 256))     
    else:
        return model
    for param in model.parameters():
        param.requires_grad = False
    # freezing params
    classifier = nn.Sequential(OrderedDict([
            ('drop1',nn.Dropout(p=0.2)),
            fc1,
            ('relu', nn.ReLU()),
            ('drop2',nn.Dropout(p=0.2)),
            fc2,
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    model.classifier = classifier

    selected_device = device("cuda" if is_available() else "cpu")
    model.to(selected_device)
    criterion = nn.NLLLoss()
    # load model parameters weights etc..
    corrupt = False
    model_state_path = save_path+'/'+"model_state_dict"
    if os.path.exists(model_state_path):
        model_state_dict = load(model_state_path)
        model.load_state_dict(model_state_dict)
        model_class_to_idx_path = save_path+'/'+"model_truths"
        if os.path.exists(model_class_to_idx_path):
            model.class_to_idx = load(model_class_to_idx_path)
            #invert dictionary to get label values based on predicted classes
            model.idx_to_labels = {v: k for k, v in model.class_to_idx.items()}
            model.eval()
        else:
            corrupt = True
    else:   
        corrupt = True
    if corrupt == True:
        return None
    else:
        return model