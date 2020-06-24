from configparser import ConfigParser

model_name = 'resnet34'
ini_pth = f'../model_config/{model_name}.ini'
cf = ConfigParser()
cf.read(ini_pth)
CUDA_VISIBLE_DEVICES = cf['DEFAULT'].get('CUDA_VISIBLE_DEVICES')
DEVICE = cf['DEFAULT']['device']
ENCODER = cf['DEFAULT']['encoder']
logdir = cf['DEFAULT']['logdir']
bs = cf.getint('train', 'bs')
epochs = cf.getint('train', 'epochs')
send = cf['train']['send']
train_csv_pth = cf['train']['train_csv_pth']
encode_lr = cf['train'].getfloat('encode_lr')
decode_lr = cf['train'].getfloat('decode_lr')
continue_train = cf.getboolean('train', 'continue_train')
patience = cf.getint('train', 'patience')
addtional = cf['train']['addtional']

