import torch
import torch.nn.functional as F
from torchvision import transforms

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return F.cross_entropy(input, target)



class ResBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cnn_layer_1 = torch.nn.Conv2d(in_c, out_c, 3, stride = 1, padding = 1)
        self.bn_layer_1 = torch.nn.BatchNorm2d(out_c)
        self.cnn_layer_2 = torch.nn.Conv2d(in_c, out_c, 3, stride = 1, padding = 1)
        self.bn_layer_2 = torch.nn.BatchNorm2d(out_c)
        self.act_layer= torch.nn.ReLU()

    def forward(self, x):
        x_dup = x
        x_new = self.cnn_layer_1(x)
        x_new = self.bn_layer_1(x_new)
        x_new = self.act_layer(x_new)
        x_new = self.cnn_layer_2(x_new)
        x_new = x_new + x_dup
        x_new = self.bn_layer_2(x_new)
        x_new = self.act_layer(x_new)
        return x_new

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """

        input_channels = 3
        img_size = 128
        in_c = img_size
        out_c = img_size
        n_classes = 6

        self.layer_1 = torch.nn.Conv2d(input_channels, out_c,3,stride=1,padding=1)
        self.layer_2 = ResBlock(in_c, out_c)
        self.layer_3 = ResBlock(in_c, out_c)
        self.layer_4 = ResBlock(in_c, out_c)
        self.layer_5 = ResBlock(in_c, out_c)
        self.output = torch.nn.Linear(in_c,n_classes)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """

        transform_norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        x = transform_norm(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x =x.mean((2,3))
        x = self.output(x)
        return x




class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        #raise NotImplementedError('FCN.__init__')

        input_channels = 3

        self.d_layer_1 = torch.nn.Sequential(
                         torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                         torch.nn.BatchNorm2d(32),
                         torch.nn.ReLU()
                         )
        self.d_layer_2 = torch.nn.Sequential(
                         torch.nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
                         torch.nn.BatchNorm2d(64),
                         torch.nn.ReLU()
                         )
        self.d_layer_3 = torch.nn.Sequential(
                         torch.nn.Conv2d(64,128, kernel_size =3, stride=1, padding=1),
                         torch.nn.BatchNorm2d(128),
                         torch.nn.ReLU()
                         )
        self.u_layer_3 = torch.nn.Sequential(
                         torch.nn.ConvTranspose2d(128,64, kernel_size=3, stride=1, padding=1),
                         torch.nn.BatchNorm2d(64),
                         torch.nn.ReLU()
                         )
        self.u_layer_2 = torch.nn.Sequential(
                         torch.nn.ConvTranspose2d(128,32, kernel_size=3, stride=1, padding=1),
                         torch.nn.BatchNorm2d(32),
                         torch.nn.ReLU()
                         )

        self.u_layer_1 = torch.nn.Sequential(
                         torch.nn.ConvTranspose2d(64,5, kernel_size=3, stride=1, padding=1),
                         torch.nn.BatchNorm2d(5),
                         torch.nn.ReLU()
                         )

        self.d_1_2 = torch.nn.ConvTranspose2d(32,64, kernel_size=3, stride=1, padding=1)
        self.d_2_3 = torch.nn.ConvTranspose2d(64,128, kernel_size=3, stride=1, padding=1)

        self.u_3_4 = torch.nn.ConvTranspose2d(128,64, kernel_size=3, stride=1, padding=1)
        self.u_4_5 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        #raise NotImplementedError('FCN.forward')
        transform_norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        x = transform_norm(x)
        s, t, h, w = x.shape
        x1 = self.d_layer_1(x) #32
        x2 = self.d_layer_2(x1) #64
        x2 = x2 + self.d_1_2(x1) #64
        x3 = self.d_layer_3(x2)  #128
        x3 = x3 + self.d_2_3(x2) #128
        x4 = self.u_layer_3(x3) #64
        x4 = x4 + self.u_3_4(x3) #64
        x4 = torch.cat([x2, x4], dim=1) #64
        x5 = self.u_layer_2(x4)  #32
        x5 = x5 + self.u_4_5(x4) #32
        x5= torch.cat([x1, x5], dim=1) #32
        x5 = self.u_layer_1(x5) #5
        x_output = x5[:,:,:h,:w]
        return x_output


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
