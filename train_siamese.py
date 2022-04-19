from siamese.siamese_model import build_siamese_model
from siamese import utils

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Lambda

from tensorflow.keras.datasets import mnist
import numpy as np

(trainX,trainY),(testX,testY)=mnist.load_data()
trainX=trainX/255.
testX=testX/255.

trainX=np.expand_dims(trainX,axis=-1)
testX=np.expand_dims(testX,axis=-1)

(pairTrain,labelTrain)=utils.make_pairs(trainX,trainY)
(pairTest,labelTest)=utils.make_pairs(testX,testY)

imgA=Input(shape=(28,28,1))
imgB=Input(shape=(28,28,1))

feat_extract=build_siamese_model((28,28,1))
featA=feat_extract(imgA)
featB=feat_extract(imgB)

distance=Lambda(utils.euclidean_distance)([featA,featB])
outputs=Dense(1,activation='sigmoid')(distance)

model=Model(inputs=[imgA,imgB],outputs=outputs)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])