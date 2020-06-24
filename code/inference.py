from torch_train_bkg import *
from preprocess import get_model
from tqdm import tqdm # roll of progress
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from train_config import *




runner = SupervisedRunner()
model,preprocess_fn = get_model(ENCODER)
train = pd.read_csv(train_csv_pth)
valid_ids = pd.read_csv(f"{path}/valids.csv")
valid_ids = valid_ids['valid_ids'].values
valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids,
                             transforms=get_validation_augmentation(),
                             preprocessing=get_preprocessing(preprocess_fn))
valid_loader = DataLoader(valid_dataset, bs, shuffle=False,num_workers=0)

loaders = {'infer':valid_loader}
runner.infer(
    model= model,
    loaders = loaders,
    callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"
        ),
        InferCallback()
    ],
    verbose=True
)


#%%
vaild_mask = []
probabilities = np.zeros((555*4,350,525))

for i,(batch,output) in enumerate(tqdm(zip(valid_dataset,runner.callbacks[0].predictions["logits"]))):
    img,mask = batch
    for j,(m,out) in enumerate(zip(mask,output)):
        if m.shape != (350,525):
            m = cv2.resize(m,dsize=(525,350),interpolation=cv2.INTER_LINEAR)
        vaild_mask.append(m)
        #
        # if out.shape != (350,525):
        #     out = cv2.resize(out, dsize=(525, 350))

    for j, prob in enumerate(output):
        if prob.shape != (350,525):
            prob = cv2.resize(prob,dsize=(525,350),interpolation=cv2.INTER_LINEAR)

        probabilities[i*4+j,:,:] = prob


#%%
class_params = {}

top = pd.DataFrame(index=['Fish','Flower','Gravel','Sugar'],columns=['threshold', 'size', 'dice'])
# top = {}
for class_id in range(4):
    print(f"-- {mask_tag[class_id]} --")
    attempts = []
    for ts in tqdm(range(0,100,5)):
        ts /= 100
        for ms in [0,100,1200,5000,10000,12000]:
            masks = []
            for i in range(class_id,len(probabilities),4):
                prob = probabilities[i]
                pred,num_pred = post_process(sigmoid(prob),ts,ms)
                masks.append(pred)

            d = []
            for i,j in zip(masks,vaild_mask[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)

                else:
                    d.append(dice(i,j))

            attempts.append((ts,ms,np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    top.iloc[class_id]=attempts_df.iloc[0]

    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    class_params[class_id] = (best_threshold, best_size)


    fig = plt.figure()
    sns.lineplot(x = 'threshold',y='dice',hue='size',data=attempts_df)
    plt.title(f'Threshold and min size vs dice for one of the class {mask_tag[class_id]}')
    # plt.show()

    plt.savefig(f"../img/{mask_tag[class_id]}_{arg.encoder}.jpg")

top.to_csv(f'{logdir}/threshold.csv')



for i, (input, output) in enumerate(zip(
        valid_dataset, runner.callbacks[0].predictions["logits"])):
    image, mask = input

    image_vis = image.transpose(1, 2, 0)
    mask = mask.astype('uint8').transpose(1, 2, 0)
    pr_mask = np.zeros((350, 525, 4))
    for j in range(4):
        probability = cv2.resize(output.transpose(1, 2, 0)[:, :, j], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        pr_mask[:, :, j], _ = post_process(sigmoid(probability), class_params[j][0], class_params[j][1])
    # pr_mask = (sigmoid(output) > best_threshold).astype('uint8').transpose(1, 2, 0)

    visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis, original_mask=mask, raw_image=image_vis,
                       raw_mask=output.transpose(1, 2, 0))

    plt.savefig(f"../img/visualize_{ENCODER}_{i}.jpg")

    if i >= 2:
        break


# save class params
with open(ini_pth,'w') as f:
    cf['test']["class_params"] = str(class_params)
    cf.write(f)