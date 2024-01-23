import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import os
import requests
from PIL import Image

nc = 3
nz = 4
ngf = 64
nef = 16

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 64 x 64``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 128 x 128``
        )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 128``
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size ``(nef) x 64 x 64``
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(nef * 2) x 32 x 32``
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(nef*4) x 16 x 16``
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(nef*8) x 8 x 8``
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(nef*16) x 4 x 4``
            nn.Flatten(),
            nn.Linear(nef*16*4*4, nz)
        )

    def forward(self, input):
        return self.main(input)

netG = Generator()
netG.load_state_dict(torch.load('netG_weights.pth', map_location=torch.device('cpu')))
netG.eval()

netE = Encoder()
netE.load_state_dict(torch.load('netE_weights.pth', map_location=torch.device('cpu')))
netE.eval()

def get_satellite_view(ll):
    params = {
        'll': ll,
        'size': '450,450',
        'z': 15,
        'l': 'sat'
    }
    return requests.get("https://static-maps.yandex.ru/1.x/",
                        params=params, verify=True)

satellites_transforms = transforms.Compose([
   transforms.Resize(360),
   transforms.CenterCrop(128),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]);


from flask import Flask, request, send_file
from PIL import Image, ImageDraw

app = Flask(__name__)

@app.route('/', methods=['GET'])
def generate_image():
   ll = request.args.get('ll')
   satellite_image_name = f'./{ll}_sat.jpg'

   response = get_satellite_view(ll)
   with open(satellite_image_name, 'wb') as f:
      f.write(response.content)

   satellite_img = Image.open(satellite_image_name, mode='r').convert('RGB')
   transformed_img = satellites_transforms(satellite_img).unsqueeze(0)

   hidden = netE(transformed_img)
   generated = netG(hidden.unsqueeze(-1).unsqueeze(-1)).detach().cpu()

   image_name = f'./{ll}.jpg';

   vutils.save_image(generated[0], image_name, normalize=True)

   return image_name



