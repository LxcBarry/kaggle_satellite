from helper_function import *
import pandas as pd
train = train = pd.read_csv('../input/train_after.csv')
image_name = '8242ba0.jpg'
image = get_img(image_name)
mask = make_mask(train,image_name)
plot_with_augmentation(image, mask, albu.GridDistortion(p=0.5))
plt.show()