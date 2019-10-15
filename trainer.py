import torch
import numpy as np

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()



        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def semantic(train_loader, model, cuda):
    classes_semantic = {}
    classes_num = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if cuda:
                data = data.cuda()
                if target is not None:
                    target = target.cuda()

            target = target.cpu().numpy()
            outputs = model(data).cpu().numpy()

            for i in range(len(data)):
                if target[i] not in classes_semantic.keys():
                    classes_semantic[target[i]] = outputs[i]
                    classes_num[target[i]] = 1
                else:
                    classes_semantic[target[i]] += outputs[i]
                    classes_num[target[i]] += 1

    for i in classes_semantic.keys():
        classes_semantic[i] = classes_semantic[i] / classes_num[i]

    return classes_semantic


def eval_precious(test_loader, classes_semantic, model, epsilon, cuda):
    test_precious = {}
    test_num = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target if len(target) > 0 else None
            if cuda:
                data = data.cuda()
                if target is not None:
                    target = target.cuda()

            target = target.cpu().numpy()
            outputs = model(data).cpu().numpy()

            for i in range(len(data)):
                minn = 1e9
                minn_target = 0
                for j in classes_semantic.keys():
                    dis = outputs[i]-classes_semantic[j]
                    tem = np.dot(dis, dis)
                    if tem < minn:
                        minn = tem
                        minn_target = j
                    #print(tem, j, target[i])
                #print(minn, minn_target, target[i])

                if (minn > epsilon and target[i] not in classes_semantic.keys()) \
                        or (minn <= epsilon and target[i] == minn_target):
                    if target[i] not in test_precious.keys():
                        test_precious[target[i]] = 1
                    else:
                        test_precious[target[i]] += 1

                if target[i] not in test_num.keys():
                    test_num[target[i]] = 1
                else:
                    test_num[target[i]] += 1

    for i in test_precious.keys():
        test_precious[i] = test_precious[i] / test_num[i]

    return test_precious