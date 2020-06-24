#%%
import configparser
cf = configparser.ConfigParser()
cf.read('../model_config/densenet121.ini')
s = cf.sections()
print("section:",s)
#%%
train_info = cf['train']
#%%
for key,val in train_info.items():
    print(f"{key}:{val}")
    print(type(val))

# #%%
# cf['test'] = {'a':1,'b':2}
#
# with open('../model_config/densenet121.ini','w') as f:
#     cf.write(f)