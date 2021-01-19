'''
BCINet
© Avinash K Singh 
https://github.com/thinknew/bcinet
Licensed under MIT License
'''

import numpy as np
import scipy.io as sio
from EEGModels import EEGNet, EEGNet_TF
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,EarlyStopping
from modelRun import *
from op import getInputDataInfo

import tensorflow.compat.v1 as tf

start=0 # Start from index 0
totalDataLength=2 # Equal to number of datasets
numOfEpochs=1
scaleFactor = 1000  # Fix parameter
numOfKernels = 1  # Fix parameter
visibleGPU="1"
patience=300
delta=0
dropoutRate=0.5
folder='test'



## EEGNet
 for i in range(start, totalDataLength,1 ):

     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                  BS, checkpointCallbacks(folder+'/EEGNet'+SaveMatFileName, patience,delta), folder+'/EEGNet'+SaveMatFileName, numOfEpochs,
              samplingRate, "EEGNet",dropoutRate,visibleGPU)

 # # DeepNet
 for i in range(start, totalDataLength,1 ):

     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)

     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                  BS, checkpointCallbacks(folder+'/DeepConvNet'+SaveMatFileName, patience,delta), folder+'/DeepConvNet'+SaveMatFileName, numOfEpochs,
              samplingRate, "DeepConvNet",dropoutRate,visibleGPU)


## Shallownet
 for i in range(start, totalDataLength,1 ):

     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                  BS, checkpointCallbacks(folder+'/ShallowConvNet'+SaveMatFileName, patience,delta), folder+'/ShallowConvNet'+SaveMatFileName, numOfEpochs,
              samplingRate, "ShallowConvNet",dropoutRate,visibleGPU)


# BCINet
for i in range(start, totalDataLength,1 ):

    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                 BS, checkpointCallbacks(folder+'/BCINet'+SaveMatFileName, patience,delta), folder+'/BCINet'+SaveMatFileName, numOfEpochs,
             samplingRate, "BCINet",dropoutRate,visibleGPU)
