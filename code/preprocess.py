import pandas as pd
# import numpy as np
import os
os.chdir(f"{os.getcwd()}/code")
from sklearn.model_selection import train_test_split
path = "../input"
print(os.listdir("../input"))
from PIL import Image
from helper_function import *
# from model import *
import segmentation_models_pytorch as smp

# data
mask_tag=['Fish','Flower','Gravel','Sugar']
def get_train_data():
    """
    :return: train_ids,valid_ids,test_ids
    """
    id_mask_count = pd.read_csv(f"{path}/id_mask_count.csv")
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values,
                                            random_state=42,
                                            stratify=id_mask_count['count'], test_size=0.1)
    sub = pd.read_csv(f"{path}/sample_submission.csv")
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    valids = pd.DataFrame({'valid_ids':valid_ids})
    valids.to_csv(f"{path}/valids.csv",index=False)
    return train_ids,valid_ids,test_ids

def get_submiss_data(sub):
    """

    :return:submisstion image id
    """

    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    return test_ids
# model
def get_model(ENCODER = 'resnet50',ENCODER_WEIGHTS = 'imagenet'):
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation= 'sigmoid'
    )
    preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model,preprocess_fn

if __name__ == "__main__":
##
#
    plot = False
    train = pd.read_csv(f"{path}/train.csv")
    sub = pd.read_csv(f"{path}/sample_submission.csv")
    # print(train.head())
    
    # get total data count
    if plot is True:
        n_train = len(os.listdir(f"{path}/train_images"))
        n_test = len(os.listdir(f"{path}/test_images"))
        print(f'There are {n_train} images in train dataset')
        print(f'There are {n_test} images in test dataset')
    
    #
    # counting by key
    train['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    
    #
    # total Effective data
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    # data distribution
    train.loc[train['EncodedPixels'].isnull()==False,'Image_Label'].apply(lambda  x:x.split('_')[0]).value_counts().value_counts()
    
    #
    # add new column
    train['label'] = train['Image_Label'].apply(lambda x:x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x:x.split('_')[0])
    
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    #
    # plot some picture with mask
    if plot is True:
        fig = plt.figure(figsize=(25, 16))
        for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):
            for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):
                ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])
                im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")
                plt.imshow(im)
                mask_rle = row['EncodedPixels']
                try: # label might not be there!
                    mask = rle_decode(mask_rle)
                except:
                    mask = np.zeros((1400, 2100))
                plt.imshow(mask, alpha=0.5, cmap='gray')
                ax.set_title(f"{row['Image_Label'].split('_')[0]}.  {row['label']}")
    
        plt.show()
    
    # create a list of unique ids for training
    ##
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False,'Image_Label'].\
        apply(lambda x:x.split('_')[0]).value_counts().\
        reset_index().rename(columns={'index':'img_id','Image_Label':'count'})
    # id_mask_count.to_csv(f"{path}/id_mask_count.csv",index=False)
    # train.to_csv(f"{path}/train_after.csv")
    sub.to_csv(f"{path}/sub_after.csv",index=False)

# get_train_data()



