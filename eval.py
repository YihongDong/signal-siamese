import sys, os
import numpy as np
import torch
from torchvision import transforms
from signaldata import SignalData, SignalDataEval

mean, std = 0, 0.1

train_dataset = SignalDataEval(train=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
classes = train_dataset.classes
test_dataset = SignalDataEval(train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))

from datasets import Siamese, SiameseEval

siamese_train_dataset = SiameseEval(train_dataset) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseEval(test_dataset)

cuda = torch.cuda.is_available()
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

from networks import EmbeddingNet, SiameseNet

embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)

if cuda:
    model.cuda()

from trainer import semantic, eval_precious

#model = torch.load('./model.pkl')
embedding_net = torch.load('./embedding.pkl')
# with torch.no_grad():
#     model.eval()
#     for batch_idx, (data, target) in enumerate(siamese_train_loader):
#         print(data[0])
#         outputs = embedding_net(data).numpy()
#         print(outputs[0])
#         break

classes_semantic = semantic(siamese_train_loader, embedding_net, cuda)
epsilon = 1e9

test_precious = eval_precious(siamese_test_loader, classes_semantic, embedding_net, epsilon, cuda)
print(test_precious)

# maxn = 0
# for epsilon in np.arange(0.01, 1, 0.01):
#     sumn = 0
#     test_precious = eval_precious(siamese_test_loader, classes_semantic, embedding_net, epsilon, cuda)
#     for i in test_precious.keys():
#         sumn += test_precious[i]
#     tem = sumn/len(classes)
#     print(tem)
#     if maxn < tem:
#         maxn = tem
#         result = epsilon
#
# print(result)
#torch.save(embedding_net, './embedding.pkl')