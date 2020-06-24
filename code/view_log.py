from configparser import ConfigParser
from catalyst.dl import utils
cf = ConfigParser()
model_name = 'inceptionresnetv2'
cf.read(f'../model_config/{model_name}.ini')
logdir = cf['DEFAULT']['logdir']
plot = True
if plot == True:
    utils.plot_metrics(
        logdir=logdir,
        metrics=["loss","dice","lr","_bace/lr"]
    )