from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Dropout,GlobalAveragePooling2D,MaxPooling2D

def build_siamese_model(inputShape,embeddingDim=48):
    
    inputs=Input(inputShape)
    x=Conv2D(64,(3,3),padding="same",activation='relu')(inputs)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Dropout(0.2)(x)
    
    x=Conv2D(64,(3,3),padding="same",activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Dropout(0.2)(x)  
    
    pooledOutput=GlobalAveragePooling2D()(x)
    outputs=Dense(embeddingDim)(pooledOutput)
    
    model=Model(inputs,outputs)
    
    return model