from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
##image preprocessing

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    width = 256
    org_width = image.size[0]
    org_height = image.size[1]
    width_percent = width / float(org_width)
    # calculate new height to preserve aspect ratio
    height = int((float(org_height) * float(width_percent)))
    image_out = image.resize((width, height), Image.ANTIALIAS)
    # set the crop box
    desired_box = 224
    left = int(width / 2 - desired_box / 2)
    upper = int(height / 2 - desired_box / 2)
    right = int(width / 2 + desired_box / 2)
    lower = int(height / 2 + desired_box / 2)
    box = (left, upper, right, lower)
    image_out = image_out.crop(box=box)
    image_out = image_out.resize((desired_box, desired_box))
    image_out = np.array(image_out, dtype=np.float)
    image_out = image_out / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_out = (image_out - mean) / std
    image_out = image_out.transpose((2, 0, 1))
    return image_out

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax