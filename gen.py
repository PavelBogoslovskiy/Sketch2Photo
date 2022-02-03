import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape

        self.fc = nn.Linear(latent_dim, self.h * self.w)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x, z):
        # Propagate noise through fc layer and reshape to img shape
        # x:(N,3,128,128) z:(N,8)
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)  # z:(N,1,128,128)

        # Сoncating (x and z): (N,4,128,128)
        d1 = self.down1(torch.cat((x, z), 1))  # d1:(N,64,64,64)
        d2 = self.down2(d1)  # d2:(N,128,32,32)
        d3 = self.down3(d2)  # d3:(N,256,16,16)
        d4 = self.down4(d3)  # d4:(N,512,8,8)
        d5 = self.down5(d4)  # d5:(N,512,4,4)
        d6 = self.down6(d5)  # d6:(N,512,2,2)
        d7 = self.down7(d6)  # d7:(N,512,1,1)
        u1 = self.up1(d7, d6)  # u1:(N,1024,2,2)
        u2 = self.up2(u1, d5)  # u2:(N,1024,4,4)
        u3 = self.up3(u2, d4)  # u3:(N,1024,8,8)
        u4 = self.up4(u3, d3)  # u4:(N,512,16,16)
        u5 = self.up5(u4, d2)  # u5:(N,256,32,32)
        u6 = self.up6(u5, d1)  # u6:(N,128,64,64)

        return self.final(u6)  # final:(N,3,128,128)