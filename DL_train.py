import torch
import os
import time
import numpy as np
from tqdm import tqdm
from torch import nn
from utils import setup_seed, nMSE
from model import FusionModel, SimpleModel, MLPFusionModel
from data import make_dataloader
def validata(val_loader, model, criterion, cuda):
    tbar = tqdm(val_loader)
    losses = 0
    num = 0
    for i, sample in enumerate(tbar):
        user, aural, visual, text = sample['user'], sample['aural'], sample['visual'], sample['text']
        target = sample['target']
        if cuda:
            user = user.cuda().float()
            visual = visual.cuda().float()
            aural = aural.cuda().float()
            text = text.cuda().float()
            target = target.cuda().float()
        with torch.no_grad():
            output = model(user, aural, visual, text)

        loss = criterion(output, target)

        losses += loss.item()
        num += 1
        tbar.set_description('val loss: %.3f' % (losses / num))
    return losses / num

def main(model_name):
    cuda = torch.cuda.is_available()
    setup_seed(20)

    if model_name == 'FusionModel':
        model = FusionModel()
    elif model_name == 'SimpleModel':
        model = SimpleModel()
    elif model_name == 'MLPFusionModel':
        model = MLPFusionModel()

    if cuda:
        model = model.cuda()

    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    path = os.path.join('./run/DL', model_name, now)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    train_loader, val_loader = make_dataloader('all', True, 100)
    test_loader = make_dataloader('all', False, 100)
    criterion = nMSE()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)

    best_val = 10
    for epoch in range(200):
        model.train()
        losses = 0
        tbar = tqdm(train_loader)
        for i, sample in enumerate(tbar):

            user, aural, visual, text = sample['user'], sample['aural'], sample['visual'], sample['text']
            target = sample['target']
            if cuda:
                user = user.cuda().float()
                visual = visual.cuda().float()
                aural = aural.cuda().float()
                text = text.cuda().float()
                target = target.cuda().float()

            output = model(user, aural, visual, text)

            loss = criterion(output, target)

            optim.zero_grad()
            loss.backward()

            optim.step()

            losses = (losses * i + loss.item()) / (i + 1)
            tbar.set_description('Train loss: %.3f' % (losses))

        val = validata(val_loader, model, criterion, cuda)
        if val < best_val:
            best_val = val
            print("current best val loss: {}".format(str(best_val)))
            output = []
            tbar = tqdm(test_loader)
            for i, sample in enumerate(tbar):

                user, aural, visual, text = sample['user'], sample['aural'], sample['visual'], sample['text']
                if cuda:
                    user = user.cuda().float()
                    visual = visual.cuda().float()
                    aural = aural.cuda().float()
                    text = text.cuda().float()
                with torch.no_grad():
                    output.append(model(user, aural, visual, text))
            result = torch.concat([output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]],
                                  dim=0).cpu().numpy()
            np.save(os.path.join(path, '2019302130055_{}_{}.npy'.format(str(epoch), str(val))), result)




if __name__ == '__main__':
    # FusionModel SimpleModel MLPFusionModel
    name = 'SimpleModel'
    main(name)