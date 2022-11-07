import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    #raise NotImplementedError('extract_peak')
    # retain the size of the heatmap: W (H) - Kernal_size + 2*padding +1 = W (H) => padding = (kernal_size-1)/2
    H, W = heatmap.shape[0], heatmap.shape[1]
    window_max, indices = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding = (max_pool_ks-1)//2, stride = 1, return_indices=True)
    peak_ind = torch.logical_and((heatmap >= window_max).float(), window_max > min_score)
    window_max = window_max[peak_ind]
    indices = indices[peak_ind]
    scores, ind = torch.topk(window_max, min(len(window_max),max_det))
    cx, cy = indices[ind]%W, indices[ind]//W
    return[*zip(scores,cx,cy)]


class Detector(torch.nn.Module):


    class Res_Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=(kernel_size-1) // 2,
                                      stride=stride, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            xx = self.c1(x)
            xx = self.b1(xx)
            xx = F.relu(xx)
            xx = self.c2(xx)
            xx = self.b2(xx)
            xx = F.relu(xx)
            xx = self.c3(xx)
            xx = self.b3(xx)
            return F.relu(xx + self.skip(x))


    class UpBlock(torch.nn.Module):
        def __init__(self, input_dim, output_dim, kernel_size=3, stride=2):
            super().__init__()
            self.up_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_dim, output_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2, stride=stride,
            output_padding = 1),
            torch.nn.ReLU()
            )

        def forward(self,x):
            return self.up_block(x)




    def __init__(self, kernel_size=3):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        #raise NotImplementedError('Detector.__init__')
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        self.res_block_1 = self.Res_Block(3,16)
        self.res_block_2 = self.Res_Block(16,32)
        self.res_block_3 = self.Res_Block(32,64)
        self.res_block_4 = self.Res_Block(64,128)

        #self.up_block_4 = self.UpBlock(128,128)
        #self.up_block_3 = self.UpBlock(128+64,64)
        #self.up_block_2 = self.UpBlock(64+32,32)
        #self.up_block_1 = self.UpBlock(32+16,16)

        self.up_block_4 = self.UpBlock(128,128)
        self.up_block_3 = self.UpBlock(128+64,64)
        self.up_block_2 = self.UpBlock(64+32,32)
        self.up_block_1 = self.UpBlock(32+16,16)

        self.output = torch.nn.Conv2d(16,3,1)


    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        #raise NotImplementedError('Detector.forward')
        x_scaled = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_blocks = []
        up_blocks.append(x_scaled)
        xx = self.res_block_1(x_scaled)
        up_blocks.append(xx)
        xx = self.res_block_2(xx)
        up_blocks.append(xx)
        xx = self.res_block_3(xx)
        up_blocks.append(xx)
        xx = self.res_block_4(xx)
        up_blocks.append(xx)
        xx = self.up_block_4(xx)
        xx = xx[:,:,:up_blocks[3].size(2), :up_blocks[3].size(3)]
        xx = torch.cat([xx,up_blocks[3]], dim=1)
        xx = self.up_block_3(xx)
        xx = xx[:,:,:up_blocks[2].size(2), :up_blocks[2].size(3)]
        xx = torch.cat([xx,up_blocks[2]], dim=1)
        xx = self.up_block_2(xx)
        xx = xx[:,:,:up_blocks[1].size(2), :up_blocks[1].size(3)]
        xx = torch.cat([xx,up_blocks[1]], dim=1)
        xx = self.up_block_1(xx)
        xx = xx[:,:,:up_blocks[0].size(2), :up_blocks[0].size(3)]
        return self.output(xx)


    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        #raise NotImplementedError('Detector.detect')
        return  [[(*peak, 0, 0) for peak in extract_peak(heatmap)] for heatmap in self(image[None]).squeeze(0)]



def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
