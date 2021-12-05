import numpy as np


visual_data = np.load('../data/mod_visual.npy')
aural_data = np.load('../data/mod_aural.npy')
social_data = np.load('../data/mod_social.npy')
textual_data = np.load('../data/mod_textual.npy')

label_data = np.load('../data/train.npy')
print('ready')