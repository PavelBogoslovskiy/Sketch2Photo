import torch
from gen import Generator
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

img_height = 128  # size of image height
img_width = 128  # size of image width
channels = 3  # number of image channels
latent_dim = 8  # number of latent codes

input_shape = (channels, img_height, img_width)  # shape of input image (tuple)

cuda = True if torch.cuda.is_available() else False  # availability of GPU
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

generator = Generator(latent_dim, input_shape)  # Initialize generator

generator.load_state_dict(torch.load('weights/generator.pth'))
generator.eval().cuda()


def image_classifier(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    img = Image.fromarray(image).convert('RGB')
    images = transform(img)
    generator.eval()

    img_A = images
    real_A = img_A.view(1, *img_A.shape).repeat(latent_dim, 1, 1, 1)
    real_A = Variable(real_A.type(Tensor))

    # Sample latent representations
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (latent_dim, latent_dim))))

    # Generate samples
    fake_B = generator(real_A, sampled_z)

    # Concatenate samples horizontally
    fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
    img_sample = torch.cat((img_A, fake_B), -1)
    img_sample = img_sample.view(1, *img_sample.shape)

    # Concatenate with previous samples vertically
    save_image(img_sample, "save.png", normalize=True)
    out = Image.open('save.png').convert('RGB')

    return out
