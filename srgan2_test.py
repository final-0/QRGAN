import torch.nn as nn
import torch
import torchvision.transforms as tr
import os
import numpy as np
from torchvision.utils import save_image, make_grid 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models2 import *
from datasets1 import *
import time

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(ImageDataset("../../data/qr_test/qr_test"), batch_size = 1, shuffle=True, num_workers = 8)

G = Generator()

if cuda:
    G = Generator().cuda()
    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_content = torch.nn.L1Loss().cuda()
"""

G = Generator()
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
"""


G.load_state_dict(torch.load("saved_models/generator.pth"))

"""""
for i, imgs in enumerate(dataloader):

    if (i+1) % 250 == 0:

        start1 = time.time()
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        gen_hr = G(imgs_lr)
        print(time.time() - start1)
        
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

        img_last = gen_hr.to('cpu').detach().numpy().copy()

        start2 = time.time()

        img = ((img_last > 0) - 0.5) *2
     
        counter = 0

        for p in range(256):
            if img[0,0,50,p] == 1:
                pass
            elif img[0,0,50,p] == -1 and img[0,0,50,p+1] == -1:
                counter += 1
            else:
                counter += 1
                break
        #print(counter)
        
        img = torch.from_numpy(img.astype(np.float32)).clone()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = img.to(device)
        trans1 = tr.Compose([tr.Resize((256//counter,256//counter), Image.NEAREST)])
        trans3 = tr.Compose([tr.Resize((256,256), Image.NEAREST)])
        img = trans1(img)
        img = trans3(img)
        print(time.time() - start2)

        img = make_grid(img, nrow=1, normalize=True)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr,gen_hr,imgs_hr,img), -1)
        save_image(img_grid, "images/%d.png" % (i+1), normalize=False)
        print(time.time()-start1)
"""
Time1 = 0
Time2 = 0
Time3 = 0

for i, imgs in enumerate(dataloader):
    
    start1 = time.time()
    imgs_lr = Variable(imgs["lr"].type(Tensor))
    imgs_hr = Variable(imgs["hr"].type(Tensor))        
    gen_hr = G(imgs_lr)
    Time1 += (time.time() - start1)
        
    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

    img_last = gen_hr.to('cpu').detach().numpy().copy()

    start2 = time.time()

    img = ((img_last > 0) - 0.5) *2
     
    counter = 0

    for p in range(256):
        if img[0,0,50,p] == 1:
            pass
        elif img[0,0,50,p] == -1 and img[0,0,50,p+1] == -1:
            counter += 1
        else:
            counter += 1
            break
    #print(counter)
        
    img = torch.from_numpy(img.astype(np.float32)).clone()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = img.to(device)
    trans1 = tr.Compose([tr.Resize((256//counter,256//counter), Image.NEAREST)])
    trans3 = tr.Compose([tr.Resize((256,256), Image.NEAREST)])
    img = trans1(img)
    img = trans3(img)
    Time2 += (time.time() - start2)

    img = make_grid(img, nrow=1, normalize=True)
    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_lr,gen_hr,imgs_hr,img), -1)
    save_image(img_grid, "images/%d.png" % (i+1), normalize=False)
    Time3 += (time.time()-start1)

print(Time1)
print(Time2)
print(Time3)
    