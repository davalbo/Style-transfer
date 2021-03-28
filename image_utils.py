from io import BytesIO
from PIL import Image
from torchvision import transforms

def load_torch_image(img_path, max_size=400, shape=None, mean=(0, 0, 0), std=(1, 1, 1)):
    
    image = Image.open(img_path).convert('RGB')
    
    if shape is not None:
        size = shape
    else:
        size = min((max(image.size), max_size))
    
    im_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = im_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

def im_convert(tensor, mean=(0,0,0), std=(1,1,1)):

    import numpy

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * numpy.array(std) + numpy.array(mean)
    image = image.clip(0, 1)

    return image