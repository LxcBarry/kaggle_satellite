import pandas as pd
from preprocess import get_submiss_data
from helper_function import *
from tqdm import tqdm
from catalyst.dl.runner import SupervisedRunner
from preprocess import get_model
from torch.utils.data import DataLoader

from configparser import  ConfigParser
import json
model_name = 'inceptionresnetv2'
cf = ConfigParser()
cf.read(f'../model_config/{model_name}.ini')
logdir = cf['DEFAULT']['logdir']

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
model,preprocess_fn = get_model(cf['DEFAULT']["encoder"])
runner = SupervisedRunner()

sub = pd.read_csv(cf['test']['sub'])

class_params = eval(cf.get('test','class_params'))

if __name__ == '__main__':
    test_ids = get_submiss_data(sub)
    test_dataset = CloudDataset(df=sub,datatype='test',img_ids=test_ids,
                                transforms=get_validation_augmentation(),preprocessing=get_preprocessing(preprocess_fn))
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)
    loader = {'vaild':test_loader}

    encoded_pixels = []
    image_id = 0

    preds = runner.predict_loader(model=model,
                          loader=test_loader,
                          resume=f'{logdir}/checkpoints/best.pth',
                          verbose=True)

    for i, output in enumerate(tqdm

                                   (preds)):
        # img, mask = batch

        for j, prob in enumerate(output):
            if prob.shape != (350, 525):
                prob = cv2.resize(prob, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(sigmoid(prob), class_params[image_id % 4][0],
                                                class_params[image_id % 4][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1
            # probabilities[i * 4 + j, :, :] = sigmoid(prob)
    # for i, test_batch in enumerate(tqdm.tqdm(loader['test'])):
    #     runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
    #     for i, batch in enumerate(runner_out):
    #         for probability in batch:
    #
    #             probability = probability.cpu().detach().numpy()
    #             if probability.shape != (350, 525):
    #                 probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
    #             predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
    #                                                 class_params[image_id % 4][1])
    #             if num_predict == 0:
    #                 encoded_pixels.append('')
    #             else:
    #                 r = mask2rle(predict)
    #                 encoded_pixels.append(r)
    #             image_id += 1

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'{logdir}/submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)