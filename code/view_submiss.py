import matplotlib.pyplot as plt
from helper_function import *
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
model_name = "inceptionresnetv2"

# read the submisstion
sub = pd.read_csv(f"../log/{model_name}/submission.csv")

sub['im_id'] = sub['Image_Label'].apply(lambda x:x.split('_')[0])
sub['label'] = sub['Image_Label'].apply(lambda x:x.split('_')[1])

fig = plt.figure(figsize=(25, 16))
for j,im_id in enumerate(tqdm(np.random.choice(sub['im_id'].unique(), 4))):
    masks = []
    img = None
    for i, (idx, row) in enumerate(sub.loc[sub['im_id'] == im_id].iterrows()):

        mask_rle = row['EncodedPixels']
        try:  # label might not be there!
            mask = rle_decode(mask_rle)
            # mask = cv2.resize(mask, dsize=(1400, 2100), interpolation=cv2.INTER_LINEAR)
        except:
            mask = np.zeros((350, 525))
        masks.append(mask.tolist())
    im = cv2.imread(f"../input/test_images/{im_id}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
    masks = np.array(masks).astype(np.uint8)
    masks = np.squeeze(masks)
    visualize(im,masks.transpose(1,2,0))



plt.savefig(f"../img/{model_name}_pred.jpg")


