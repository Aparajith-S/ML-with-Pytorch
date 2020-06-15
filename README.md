# ML-with-Pytorch
Deep learning exercises with Pytorch and torchvision
In this project, a code for an image classifier is built with PyTorch then converted into a command line application.

## Environment
The command line application is built using python 3.6 and pytorch version   
Image operations and data augmentation are done using torchvision library


##Running the Application
###Training the model
`train.py` builds a model using default or user preferred hyperparameters.

use the following bare minimum command to run the code  

    $ python train.py data_directory --gpu 

where `data_directory` is the `flowers/` data-set root directory.
Use the -h flag to bring up the argument list help menu which displays the following.  
    
    usage: 
    
    train.py [-h] [-s SAVE] [-a ARCH] [-r LEARNING_RATE] [-d HIDDEN_UNITS]
                [-e EPOCHS] [-p]
                data_dir

    training application

    positional arguments:
  
    data_dir              data dir for images

    optional arguments:  
  
    -h, --help            show this help message and exit  
    -s SAVE, --save SAVE  Save dir for checkpoint  
    -a ARCH, --arch ARCH  architecture to be used : default is vgg16_bn. other  
                          models are alexnet and vgg13_bn  
    -r LEARNING_RATE, --learning_rate LEARNING_RATE  
                          learning rate for the model  
    -d HIDDEN_UNITS, --hidden_units HIDDEN_UNITS  
                          number of hidden units, default is 4096  
    -e EPOCHS, --epochs EPOCHS  
                          epochs for the model. default is 20  
    -p, --gpu             add this flag to enable gpu if available
    
###Running the inference 
`predict.py` runs the inference of the trained model which requires two bare 
minimum parameters `image_path` and `saved_model_checkpoint` in that order.  
One example of running this is using the following command.

    $ python predict.py flowers/test/14/image_06083.jpg vgg16_bn --gpu 

Note that the gpu flag is optional.  

Again, using the help command is a good way to find what the application accepts as arguments
    
    $ python predict.py -h

    usage: predict.py [-h] [-t TOP_K] [-c CATEGORY_NAMES] [-p] img_file checkpoint

    Inference Application

    positional arguments:
        img_file              Filepath of image
        checkpoint            Checkpoint name, enter the model name such as
                        vgg16_bn, vgg13_bn or alexnet

    optional arguments:
        -h, --help            show this help message and exit
        -t TOP_K, --top_k TOP_K
                              Number of top classes to be displayed. default is 3
        -c CATEGORY_NAMES, --category_names CATEGORY_NAMES
                              Path to the json file with category names
        -p, --gpu             Add this flag to enable gpu if available

##Code

`train.py` contains 

- `loadData(data_dir,batch_size=64)` loads the image data and does preprocessing and augmentation on it  
arguments:  
  - `data_dir` data directory containing the images
  - `batch_size` for training images. default is 64  
returns: 
  - `image_datasets` that are augmented images using rotational, flip transforms and are
   resized and center cropped to a dimension of 224x224. Normalization of the image data is done using the mean and stddev values for the three color channels.
  - `dataLoaders` that are of type `torchvision.utils.data.DataLoader` contains the dataloaders for training , validation and test datasets.

- `trainingPipeline(model,params,dataloaders)`  
arguments  
  - `model` which is the untrained newly constructed model
  - `dataloaders` accepts the dataloaders to train and validate the model  
returns  
  - `model` which is the trained model 
  
- `testModel(model,params,testloader)` tests the model and displays the test accuracy  
arguments
  - `model` which is the trained model
  - `params` which is the user configured parameters sent as arguments to `train.py` application.  
  - `testloader` a part of dataloader stored under 'test' section whose labels were not used during training phase  
return  
  - `top_p` predicted class probabilities of the test dataset
  - `top_class` predicted classes of the test dataset

- `saveCheckpoint(model,image_datasets,params)` saves the model weights, label indices and optimizer state dictionary   
arguments
    - `model` the trained model to be saved
    - `image_datasets` the image_dataset from which label index to class relation is derived
    - `params` user params that need to be saved as well as to get the save folder entered by the user/default  
return
    - `None`  
    
`modelparameters.py`contains 
- `constructModel(params)` returns constructed `model` using the user defined hyperparameters and architecture passed
   through as argument as `params`  

- `load_checkpoint (save_path,params)` returns a constructed model using the saved model artifacts such as the state 
dictionary in the `save_path` and user parameters `params`.  

`preprocessing.py` contains 
- `process_image(image)` which takes in a PIL image and crops, resizes it for the trained model in Pytorch. 
It returns the normalized image as a `numpy` array.
- `imshow(image, ax=None, title=None)` additional image show function that is useful while displaying 
using a standalone application 

`utils.py` contains
- `getLabelDict(filename)` returns a dictionary with class name label value pairs from the passed `filename`.

`predict.py` contains 
- `predict(image_path, model, topk=5)` which makes an inference using the trained model loaded using `load_checkpoint(...)` 
from `modelparameters.py`
    - `image_path` the pathname of a test image relative to the root of `predict.py`  
    - `model` loaded model with checkpoint
    - `topk` the top k predicted classes to be returned along with their probabilities  
    returns
    - `top_p` and `top_class` lists.
- `display(filename,model,probs,classes)` displays onto the console the top `k` predicted class names and their predicted probabilities
 given by `--top_k` attribute 
