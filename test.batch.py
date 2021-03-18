import requests
import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image

from u2net import U2NETP

# weights can be downoloaded here
# https://github.com/xuebinqin/U-2-Net 
model_file = "weights/u2netp.pth"
filename = "img/test.jpg"
outfile = "img/test.u2netp.png"
input_size = 320

# load image from file or url
def load_image(file_or_url_or_path):
    if isinstance(file_or_url_or_path, str) and file_or_url_or_path.startswith("http"):
        file_or_url_or_path = requests.get(file_or_url_or_path, stream=True).raw
    return Image.open(file_or_url_or_path)


# convert image for use in model pillow -> torch
def convert_image(pillow_image):

    # resize & convert to rgb
    image = pillow_image.resize((input_size, input_size))
    image = image.convert('RGB')
    # pillow -> numpy
    image = np.array(image)
    # convert to LAB 
    tmpImg = np.zeros((image.shape[0], image.shape[1],3))
    # normalize
    image = image/np.max(image) 
    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
    # reshape (320,320,3) -> (3,320,320)
    tmpImg = tmpImg.transpose((2, 0, 1))
    # reshape (3,320,320)  -> (1, 3, 320, 320)
    tmpImg = tmpImg[np.newaxis,:,:,:]
    # tmpImg = np.concatenate((tmpImg, tmpImg))
    # numpy -> torch
    image = torch.from_numpy(tmpImg)
    image = image.type(torch.FloatTensor)
    image = Variable(image)
    
    return image


# Normalize tensor (torch version)
def normalize(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


# save results
def save_output(image, mask, out_name):
    # normalize (-#,+#) -> (0.0,1.0)
    mask = normalize(mask)
    # reshape (1,320,320) -> (320,320)
    mask = mask.squeeze()
    # sacle (0.0, 1.0) -> (0, 255)
    mask = mask.cpu().data.numpy()*255
    # numpy -> pillow
    mask = Image.fromarray(mask).convert("L")
    # scale mask to original
    mask = mask.resize(image.size, resample=Image.BILINEAR)
    # convert source to RGBA
    image.convert('RGBA')
    # alpha from mask
    image.putalpha(mask)
    # save
    image.save(out_name)


def main():

    # init model
    net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    net.eval()
    
    # load image
    pillow_image_01 = load_image('img/test_01.jpg')
    torch_image_01 = convert_image(pillow_image_01)
    
    pillow_image_02 = load_image('img/test_02.jpg')
    torch_image_02 = convert_image(pillow_image_02) 

    torch_images = torch.cat((torch_image_01, torch_image_02))
    print(torch_images.shape)
    
    # feed to model
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7 = net(torch_images)

    # recieve d1 mask
    mask = d1[0,0,:,:]
    save_output(pillow_image_01, mask, f"img/01.png")
    mask = d1[1,0,:,:]
    save_output(pillow_image_02, mask, f"img/02.png")

    # cleanup
    del d1,d2,d3,d4,d5,d6,d7

main()
