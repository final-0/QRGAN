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
from torchsummary import summary
import time

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(ImageDataset("../../data/qr_dataset"), batch_size = 1, shuffle=False, num_workers = 8)

if cuda:
    G = Generator().cuda()
    D = Discriminator((1,256,256)).cuda()
    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_content = torch.nn.L1Loss().cuda()

summary(D,(1,256,256))
summary(G,(1,64,64))

optimizer_G = torch.optim.Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), 0.0002, betas=(0.5, 0.999))
print(918)
for i, imgs in enumerate(dataloader):

    imgs_lr = Variable(imgs["lr"].type(Tensor))
    imgs_hr = Variable(imgs["hr"].type(Tensor))
    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *D.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *D.output_shape))), requires_grad=False)


    #  Train Generators

    optimizer_G.zero_grad()
    gen_hr = G(imgs_lr)

    loss_GAN = criterion_GAN(D(gen_hr), valid) + criterion_GAN(D(imgs_hr), fake)
    #loss_x = criterion_GAN(D(gen_hr),fake) + criterion_GAN(D(imgs_hr), valid)
    loss_content = criterion_content(gen_hr, imgs_hr)
    # Total loss
    loss_G = 1e-3 * (loss_GAN) + loss_content
    loss_G.backward()
    optimizer_G.step()


    #  Train Discriminator

    optimizer_D.zero_grad()
    loss_real = criterion_GAN(D(imgs_hr), valid)
    loss_fake = criterion_GAN(D(gen_hr.detach()), fake)
    #loss_y = criterion_GAN(D(G(imgs_lr)), valid) + criterion_GAN(D(imgs_hr), fake)
    #loss_content1 = criterion_content(G(imgs_lr), imgs_hr)
    # Total loss
    loss_D = (loss_real + loss_fake)
    loss_D.backward()
    optimizer_D.step()
        
    #print("[Batch %d/%d] [D loss: %f] [G loss: %f]" % (i, len(dataloader), loss_D.item(), loss_G.item()))

    if (i+1) % 250 == 0:
        
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

        img_last = gen_hr.to('cpu').detach().numpy().copy()
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
        print(counter)
        counter = 0
        for q in range(256):
            if img[0,0,52,q] == 1:
                pass
            elif img[0,0,52,q] == -1 and img[0,0,52,q+1] == -1:
                counter += 1
            else:
                counter += 1
                break
        print(counter)
        img = torch.from_numpy(img.astype(np.float32)).clone()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = img.to(device)
        trans1 = tr.Compose([tr.Resize((256//counter,256//counter), Image.NEAREST)])
        trans3 = tr.Compose([tr.Resize((256,256), Image.NEAREST)])
        img = trans1(img)
        img = trans3(img)

        img = make_grid(img, nrow=1, normalize=True)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr,gen_hr,imgs_hr), -1)
        save_image(img_grid, "images/%d.png" % (i+1), normalize=False)


for i, imgs in enumerate(dataloader):

    if (i+1) % 75 == 0:

        start = time.time()

        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
    
        gen_hr = G(imgs_lr)

        print(time.time() - start)
        
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

        img_last = gen_hr.to('cpu').detach().numpy().copy()
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
        print(counter)
        counter = 0
        for q in range(256):
            if img[0,0,52,q] == 1:
                pass
            elif img[0,0,52,q] == -1 and img[0,0,52,q+1] == -1:
                counter += 1
            else:
                counter += 1
                break
        print(counter)
        img = torch.from_numpy(img.astype(np.float32)).clone()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = img.to(device)
        trans1 = tr.Compose([tr.Resize((256//counter,256//counter), Image.NEAREST)])
        trans3 = tr.Compose([tr.Resize((256,256), Image.NEAREST)])
        img = trans1(img)
        img = trans3(img)

        img = make_grid(img, nrow=1, normalize=True)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr,gen_hr,imgs_hr,img), -1)
        save_image(img_grid, "images/a%d.png" % (i+1+1), normalize=False)

torch.save(G.state_dict(), "saved_models/generator.pth")
torch.save(D.state_dict(), "saved_models/discriminator.pth")
