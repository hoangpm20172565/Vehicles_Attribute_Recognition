import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
    'car' : 'car_attr',
}
num_cls_dict = { 'market':30, 'duke':23 , 'car':20}
num_ids_dict = { 'market':751, 'duke':702 , 'car':0}

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to test image')
parser.add_argument('--dataset', default='car', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke', 'car']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]


######################################################################
# Model and Data
# ---------
def load_network(network):
    checkpoint_path = '/home/minh/Documents/AI/checkpoints'
    save_path = os.path.join(checkpoint_path, model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src


model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
model.eval()

src = load_image(args.image_path)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('/home/minh/Documents/AI/inference/details/labels.json', 'r') as f:    # load the label
            self.label_list = json.load(f)['labels']
        with open('/home/minh/Documents/AI/inference/details/attributes.json', 'r') as f:    # load the detail description pf all attributes
            self.attribute_dict = json.load(f)['attributes']
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        # print(self.attribute_dict)
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))

        fig, ax = plt.subplots()
        img = mpimg.imread(args.image_path)
        ax.imshow(img)
        name = args.image_path.split('/')[-1]
        ax.title.set_text(f'{name}')
        plt.show()


if not args.use_id:
    out = model.forward(src)
else:
    out, _ = model.forward(src)

pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5 (using greater than -> gt)

Dec = predict_decoder(args.dataset)
Dec.decode(pred)
