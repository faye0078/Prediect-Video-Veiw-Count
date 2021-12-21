from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

class XiguaData(Dataset):
    def __init__(self, type, train):
        self.type = type
        self.train = train
        if train:
            if type == 'user':
                self.user = np.load('../data/mod_social.npy')
                self.target = np.load('../data/train.npy')
                self.length = self.user.shape[0]

            elif type == 'media':
                self.aural = np.load('../data/mod_aural.npy')
                self.visual = np.load('../data/mod_visual.npy')
                self.text = np.load('../data/mod_textual.npy')
                self.target = np.load('../data/train.npy')
                self.length = self.aural.shape[0]

            elif type == 'all':
                self.user = np.load('../data/mod_social.npy')
                self.aural = np.load('../data/mod_aural.npy')
                self.visual = np.load('../data/mod_visual.npy')
                self.text = np.load('../data/mod_textual.npy')
                self.target = np.load('../data/train.npy')
                self.length = self.aural.shape[0]

        else:
            if type == 'user':
                self.user = np.load('../data/mod_social_test.npy')
                self.length = self.user.shape[0]

            elif type == 'media':
                self.aural = np.load('../data/mod_aural_test.npy')
                self.visual = np.load('../data/mod_visual_test.npy')
                self.text = np.load('../data/mod_textual_test.npy')

            elif type == 'all':
                self.user = np.load('../data/mod_social_test.npy')
                self.aural = np.load('../data/mod_aural_test.npy')
                self.visual = np.load('../data/mod_visual_test.npy')
                self.text = np.load('../data/mod_textual_test.npy')
                self.length = self.aural.shape[0]

    def __getitem__(self, item):
        if self.train:
            if self.type == 'user':
                return {'user': self.user[item],
                        'target': self.target[item]}
            elif self.type == 'media':
                return {'aural': self.aural[item],
                        'visual': self.visual[item],
                        'text': self.text[item],
                        'target': self.target[item]}
            elif self.type == 'all':
                return {'user': self.user[item],
                        'aural': self.aural[item],
                        'visual': self.visual[item],
                        'text': self.text[item],
                        'target': self.target[item]}

        else:
            if self.type == 'user':
                return {'user': self.user[item]}
            elif self.type == 'media':
                return {'aural': self.autral[item],
                        'visual': self.visual[item],
                        'text': self.text[item],}
            elif self.type == 'all':
                return {'user': self.user[item],
                        'aural': self.aural[item],
                        'visual': self.visual[item],
                        'text': self.text[item]}

    def __len__(self):
        return self.length

def make_dataloader(type='user', train=True, batch_size=128):
    dataset = XiguaData(type, train)
    if train:
        n_examples = len(dataset)
        n_train = int(n_examples * 0.8)
        train_set, val_set = random_split(dataset, [n_train, n_examples - n_train])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        return train_loader, val_loader
    else:
        test_set = XiguaData(type, train)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        return test_loader

if __name__ =='__main__':
    train_loader, val_loader = make_dataloader('media', True, 128)
    for (i, sample) in enumerate(train_loader):
        user = sample['text']
        print(user)