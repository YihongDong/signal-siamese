import sys, os
import numpy as np
import torch
from torchvision import transforms
from signaldata import SignalData

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

mean, std = 0, 0.1

train_dataset = SignalData(train=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
classes = train_dataset.classes
test_dataset = SignalData(train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))


# Set up data loaders
# from datasets import Siamese
#
# siamese_train_dataset = Siamese(train_dataset) # Returns pairs of images and target same/different
# siamese_test_dataset = Siamese(test_dataset)

from datasets import Triplet

train_dataset = Triplet(train_dataset) # Returns triplets of images
test_dataset = Triplet(test_dataset)
#print(siamese_train_dataset.__getitem__(0)[0][0][0])

cuda = torch.cuda.is_available()
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet, TripletNet
from losses import TripletLoss, ContrastiveLoss
import torch.optim as optim
from torch.optim import lr_scheduler

margin = 1.
embedding_net = EmbeddingNet()
#model = SiameseNet(embedding_net)
model = TripletNet(embedding_net)

if cuda:
    model.cuda()
#loss_fn = ContrastiveLoss(margin)
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100


from trainer import fit

#model = torch.load('./model.pkl')
#embedding_net = torch.load('./embedding.pkl')
#state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':n_epochs}
fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

#torch.save(state, './parameters.pkl')
torch.save(model, './model.pkl')
torch.save(embedding_net, './embedding.pkl')