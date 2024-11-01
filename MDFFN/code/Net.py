
import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet_Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Down, self).__init__()

        self.Down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # reduced inception (RI)
        out = self.Down(x)
        return out


class Unet_Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Up, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Up(x)


class Channel_Attention(nn.Module):
    def __init__(self, in_ch):
        super(Channel_Attention, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_ch // 2, out_channels=in_ch, kernel_size=1), nn.ReLU())
        self.sig = nn.Sigmoid()

    def forward(self, x):
        average = self.conv2(self.conv1(self.ap(x)))
        max = self.conv2(self.conv1(self.mp(x)))
        return x * self.sig(max + average)


class Spatial_Attention(nn.Module):


    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()  # shape = [32, 64, 2000, 80]

        y = self.avg_pool(X_input)  # shape = [32, 64, 1, 1]
        y = y.view(b, c)  # shape = [32,64]

        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)  # shape = [32, 64] * [64, 4] = [32, 4]

        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y)  # shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)  # shape = [32, 64, 1, 1]， 这个就表示上面公式的s, 即每个通道的权重

        return X_input * y.expand_as(X_input)

class MulAttontion(nn.Module):
    def __init__(self, norm_flag='l2'):
        super(MulAttontion, self).__init__()
        self.MHA5=MultiHeadAttention(64,8,8)
        self.MHA7=MultiHeadAttention(128,8,16)
        self.MHA9=MultiHeadAttention(256,8,32)


    def forward(self, x):
         B, C, H, W = x.size()
         if H==9:
             x_=self.MHA5(x)
         elif H==7:
             x_=self.MHA7(x)
         else:
             x_=self.MHA9(x)
         return  x+x_
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads, dim_head):
        super(MultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.total_dim = num_heads * dim_head
        self.gamma = nn.Parameter(torch.zeros(1))

        self.query = nn.Conv2d(in_channels, self.total_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.total_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels, self.total_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(self.total_dim, in_channels, kernel_size=1)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Shape: (batch_size, num_heads, height * width, dim_head)
        query = self.query(x).view(batch_size, self.num_heads, self.dim_head, height * width).permute(0, 1, 3, 2)
        key = self.key(x).view(batch_size, self.num_heads, self.dim_head, height * width).permute(0, 1, 3, 2)
        value = self.value(x).view(batch_size, self.num_heads, self.dim_head, height * width).permute(0, 1, 3, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value).permute(0, 1, 3, 2).contiguous().view(batch_size, self.total_dim, height, width)

        out = self.out_proj(out)
        out = self.gamma * out + x

        return out

class Net(nn.Module):
    def __init__(self, in_channel):
        super(Net, self).__init__()
        self.MA=MulAttontion()
        self.conv0_0 = nn.Conv2d(in_channel, 64, 1)
        self.down0_1 = Unet_Down(64, 128)
        self.down1_2 = Unet_Down(128, 256)
        self.down2_3 = Unet_Down(256, 512)

        self.up3_0 = Unet_Up(512, 256)
        self.up2_1 = Unet_Up(256, 128)
        self.up1_2 = Unet_Up(128, 64)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(192, 64, 1)
        self.conv2 = nn.Conv2d(384, 128, 1)
        self.conv3 = nn.Conv2d(768, 256, 1)
        self.conv1_1 = nn.Conv2d(64, 128, 3)

        self.conv2_2 = nn.Conv2d(384, 128, 1)
        self.se1 = SE_Block(64)#通道大小不变
        self.se2 = SE_Block(128)
        self.se3 = SE_Block(256)
        self.ca1 = Channel_Attention(64)
        self.ca2 = Channel_Attention(128)
        self.ca3 = Channel_Attention(256)
        self.sa = Spatial_Attention()#CBAM中的空间注意力
        self.fc = nn.Sequential(
            nn.Linear(6272, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x_ = self.conv0_0(x)# 64 9
        x_down1 = self.down0_1(x_) # 128 7
        x_down2 = self.down1_2(x_down1) # 256 5
        y_ = self.conv0_0(y)
        y_down1 = self.down0_1(y_)
        y_down2 = self.down1_2(y_down1)
        diff1 = x_ - y_   # 64 9
        diff2 = x_down1 - y_down1 # 128 7
        diff3 = x_down2 - y_down2 # 256 5

        x_ = self.se1(x_)
        y_down1 = self.se2(y_down1)
        x_down2 = self.se3(x_down2)
        y_ = self.se1(y_)
        y_down1 = self.se2(y_down1)
        y_down2 = self.se3(y_down2)
        diff1 = diff1 * self.se1(self.sa(self.ca1(diff1)))
        diff2 = diff2 * self.se2(self.sa(self.ca2(diff2)))
        diff3 = diff3 * self.se3(self.sa(self.ca3(diff3)))
        diff1 = self.conv1(torch.cat((x_, y_, diff1), dim=1))
        diff2 = self.conv2(torch.cat((x_down1, y_down1, diff2), dim=1))
        diff3 = self.conv3(torch.cat((x_down2, y_down2, diff3), dim=1))
        diff1=self.MA(diff1)
        diff2=self.MA(diff2)
        diff3=self.MA(diff3)

        diff1 = self.conv1_1(diff1)
        diff3 = self.up2_1(diff3)
        diff2_ = torch.cat((diff1, diff2, diff3), dim=1)
        diff2_ = self.conv2_2(diff2_)#128*3---128
        out = diff1 + diff2 + diff2_ + diff3
        out_ = torch.flatten(out, 1, 3)
        out_ = self.fc(out_)
        out__ = self.softmax(out_)
        return out__