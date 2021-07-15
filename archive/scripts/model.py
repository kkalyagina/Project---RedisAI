import numpy as np
import redisai as rai
import albumentations as A
import os, sys
import datetime, random, time
import pickle

from RedMask.utils.mask_label import mask_label
from RedMask.utils.rediai_func import modelrun tensorget tensorset

#Getting arguments from docker-compose
if len(sys.argv) != 4 :
    sys.stderr.write("Arguments error. Prefix, frequency (pics per minute) and model (tflite_model or tf_model)\n")
    sys.exit(1)

script_prefix = sys.argv[1]
pause_dur = 60. / float(sys.argv[2])
model=sys.argv[3]

#Getting tensor list
with open('data/tensors/tensorlist.pkl', 'rb') as tensors_pkl:
    images = pickle.load(tensors_pkl)

#Connection to RedisAI
con = rai.Client(host='redisai', port=6379)

#Amount of scores for saving to file 
unsaved_scores = int(sys.argv[2]) / 2

#Endless cycle of stress testing
i = 0
time_scores = []
while True:
#for i in range(100):
    starttime = datetime.datetime.now()
    con.tensorset('img'+str(script_prefix)+str(i), np.array([random.choice(images)]), dtype='float32')
    con.modelrun(model, ['img'+str(script_prefix)+str(i)], ['out_'+str(script_prefix)])
    output = con.tensorget('out_'+str(script_prefix))
    #print(output)
    #Removing tensors
    con.delete('img'+str(script_prefix)+str(i))
    con.delete('out_'+str(script_prefix))
    #Getting and saving scores
    time_scores.append(str(script_prefix)+','+str(i)+','+str((datetime.datetime.now() - starttime).total_seconds())+'\n')
    if len(time_scores) >= unsaved_scores:
        with open('scores/times_tflite.txt','a') as times:
            times.writelines(time_scores)
        time_scores = []
    time.sleep(pause_dur)
    i += 1

#for i in range(len(output)):
    #idb = np.where(output[i] == output[i].max())[0][0]
    #print (images_names[i], ' - ', mask_label[idb])
