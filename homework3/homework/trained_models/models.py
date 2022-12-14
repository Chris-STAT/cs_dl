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
        self.cnn_layer = torch.nn.Conv2d(in_c, out_c, 3, stride = 1, padding = 1)
        self.bn_layer = torch.nn.BatchNorm2d(out_c)
        self.act_layer = torch.nn.ReLU()

    def forward(self, x):
        x_dup = x
        x_new = self.cnn_layer(x)
        x_new = self.bn_layer(x_new)
        x_new = self.act_layer(x_new)
        x_new = x_new + x
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
        img_size = 64
        in_c = img_size
        out_c = img_size
        n_classes = 6

        self.layer_1 = torch.nn.Conv2d(input_channels, out_c,3,stride=1,padding=1)
        self.layer_2 = ResBlock(in_c, out_c)
        self.layer_3 = ResBlock(in_c, out_c)
        self.layer_4 = ResBlock(in_c, out_c)
        #self.layer_5 = torch.nn.Sequential(
        #                                   torch.nn.Conv2d(32,32,3, stride=1, padding=1),
        #                                   torch.nn.ReLU(),
        #                                   torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #                                   torch.nn.Dropout(p=0.1)
        #                                   )
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
        #x = self.layer_5(x)
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
        raise NotImplementedError('FCN.__init__')

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
        raise NotImplementedError('FCN.forward')


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
