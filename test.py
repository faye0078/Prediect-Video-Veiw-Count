import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from utils import setup_seed
from model import FusionModel
from data import make_dataloader
def main():
    cuda = torch.cuda.is_available()
    setup_seed(6432)

    model = FusionModel()
    if cuda:
        model = model.cuda()
    a = torch.load('save_1_0.7703858971595764.pt')
    model.load_state_dict(a)
    test_loader = make_dataloader('all', False, 100)

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
    result = torch.concat([output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]], dim=0).cpu().numpy()
    np.save('2019302130055.npy', result)



if __name__ == '__main__':
    main()