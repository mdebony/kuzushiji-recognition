from tensorflow import keras

def trainAndEvaluateModel(model, filenameModel, batchSize, inputData, outputData, nVal, nTest, lossFunc='sparse_categorical_crossentropy', maxEpochs=75):

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filenameModel,
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='auto', period=1)
    history = keras.callbacks.History()
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1, patience=2,
                                                 verbose=0,
                                                 mode='auto')
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=1e-7, patience=5,
                                              verbose=0, mode='auto')
    callbacks = [checkpoint, history, reduceLR, earlyStop]

    model.compile(optimizer='adam', loss=lossFunc, metrics=['accuracy'])
#    model.summary() #DeleteLater
    model.fit(inputData[:nVal], outputData[:nVal], epochs=maxEpochs, batch_size=batchSize,
              validation_data=(inputData[nVal:nTest], outputData[nVal:nTest]), shuffle=True,
              callbacks=callbacks, verbose=1)

    model = keras.models.load_model(filenameModel)
    test_loss, test_acc = model.evaluate(inputData[nVal:nTest], outputData[nVal:nTest])

    return test_loss

def createKMNISTModel1(inputShape, outputLength, dropoutRate, convLayer, denseLayer, kernelSize = (3,3), finalActivationFunction = 'softmax'):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(convLayer, kernelSize, activation='relu', input_shape=inputShape))
    model.add(keras.layers.SpatialDropout2D(dropoutRate))
    model.add(keras.layers.Conv2D(convLayer, kernelSize, activation='relu'))
    model.add(keras.layers.SpatialDropout2D(dropoutRate))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(convLayer*2, kernelSize, activation='relu'))
    model.add(keras.layers.SpatialDropout2D(dropoutRate))
    model.add(keras.layers.Conv2D(convLayer*2, kernelSize, activation='relu'))
    model.add(keras.layers.SpatialDropout2D(dropoutRate))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(denseLayer, activation='relu'))
    model.add(keras.layers.Dropout(dropoutRate))
    model.add(keras.layers.Dense(outputLenght, activation=finalActivationFunction))
    
    return model

def createModelUNet(inputShape, dropoutRate, convLayer):
    inputLayer = keras.Input(inputShape, name='input')
    #1st Stage
    conv1a = keras.layers.Conv2D(convLayer, (3, 3), activation='relu', padding = 'same', name='conv1a')(inputLayer)
    drop1a = keras.layers.SpatialDropout2D(dropoutRate, name='drop1a')(conv1a)
    conv1b = keras.layers.Conv2D(convLayer, (3, 3), activation='relu', padding='same', name='conv1b')(drop1a)
    drop1b = keras.layers.SpatialDropout2D(dropoutRate, name='drop1b')(conv1b)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(drop1b)
    
    #2nd Stage
    conv2a = keras.layers.Conv2D(convLayer*2, (3, 3), activation='relu', padding = 'same', name='conv2a')(pool1)
    drop2a = keras.layers.SpatialDropout2D(dropoutRate, name='drop2a')(conv2a)
    conv2b = keras.layers.Conv2D(convLayer*2, (3, 3), activation='relu', padding = 'same', name='conv2b')(drop2a)
    drop2b = keras.layers.SpatialDropout2D(dropoutRate, name='drop2b')(conv2b)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(drop2b)
    
    #3rd Stage
    conv3a = keras.layers.Conv2D(convLayer*4, (3, 3), activation='relu', padding = 'same', name='conv3a')(pool2)
    drop3a = keras.layers.SpatialDropout2D(dropoutRate, name='drop3a')(conv3a)
    conv3b = keras.layers.Conv2D(convLayer*4, (3, 3), activation='relu', padding = 'same', name='conv3b')(drop3a)
    drop3b = keras.layers.SpatialDropout2D(dropoutRate, name='drop3b')(conv3b)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(drop3b)
    
    #4th Stage
    conv4a = keras.layers.Conv2D(convLayer*8, (3, 3), activation='relu', padding = 'same', name='conv4a')(pool3)
    drop4a = keras.layers.SpatialDropout2D(dropoutRate, name='drop4a')(conv4a)
    conv4b = keras.layers.Conv2D(convLayer*8, (3, 3), activation='relu', padding = 'same', name='conv4b')(drop4a)
    drop4b = keras.layers.SpatialDropout2D(dropoutRate, name='drop4b')(conv4b)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4b)
    
    #5th Stage
    conv5a = keras.layers.Conv2D(convLayer*16, (3, 3), activation='relu', padding = 'same', name='conv5a')(pool4)
    drop5a = keras.layers.SpatialDropout2D(dropoutRate, name='drop5a')(conv5a)
    conv5b = keras.layers.Conv2D(convLayer*16, (3, 3), activation='relu', padding = 'same', name='conv5b')(drop5a)
    drop5b = keras.layers.SpatialDropout2D(dropoutRate, name='drop5b')(conv5b)
    
    #6th Stage
    up6 = keras.layers.Conv2DTranspose(convLayer*8, (3, 3), strides=(2, 2), activation='relu', padding = 'same', name='conv6a')(drop5b)    
    merge6 = keras.layers.concatenate([drop4b, up6], axis=3, name='merge6')
    conv6b = keras.layers.Conv2D(convLayer*8, (3, 3), activation='relu', padding = 'same', name='conv6b')(merge6)
    drop6b = keras.layers.SpatialDropout2D(dropoutRate, name='drop6b')(conv6b)
    conv6c = keras.layers.Conv2D(convLayer*8, (3, 3), activation='relu', padding = 'same', name='conv6c')(drop6b)
    drop6c = keras.layers.SpatialDropout2D(dropoutRate, name='drop6c')(conv6c)

    #7th Stage
    up7 = keras.layers.Conv2DTranspose(convLayer*4, (3, 3), strides=(2, 2), activation='relu', padding = 'same', name='conv7a')(drop6c) 
    merge7 = keras.layers.concatenate([drop3b, up7], axis=3, name='merge7')
    conv7b = keras.layers.Conv2D(convLayer*4, (3, 3), activation='relu', padding = 'same', name='conv7b')(merge7)
    drop7b = keras.layers.SpatialDropout2D(dropoutRate, name='drop7b')(conv7b)
    conv7c = keras.layers.Conv2D(convLayer*4, (3, 3), activation='relu', padding = 'same', name='conv7c')(drop7b)
    drop7c = keras.layers.SpatialDropout2D(dropoutRate, name='drop7c')(conv7c)
    
    #8th Stage
    up8 = keras.layers.Conv2DTranspose(convLayer*2, (3, 3), strides=(2, 2), activation='relu', padding = 'same', name='conv8a')(drop7c) 
    merge8 = keras.layers.concatenate([drop2b, up8], axis=3, name='merge8')
    conv8b = keras.layers.Conv2D(convLayer*2, (3, 3), activation='relu', padding = 'same', name='conv8b')(merge8)
    drop8b = keras.layers.SpatialDropout2D(dropoutRate, name='drop8b')(conv8b)
    conv8c = keras.layers.Conv2D(convLayer*2, (3, 3), activation='relu', padding = 'same', name='conv8c')(drop8b)
    drop8c = keras.layers.SpatialDropout2D(dropoutRate, name='drop8c')(conv8c)
    
    #9th Stage
    up9 = keras.layers.Conv2DTranspose(convLayer, (3, 3), strides=(2, 2), activation='relu', padding = 'same', name='conv9a')(drop8c) 
    merge9 = keras.layers.concatenate([drop1b, up9], axis=3, name='merge9')
    conv9b = keras.layers.Conv2D(convLayer, (3, 3), activation='relu', padding = 'same', name='conv9b')(merge9)
    drop9b = keras.layers.SpatialDropout2D(dropoutRate, name='drop9b')(conv9b)
    conv9c = keras.layers.Conv2D(convLayer, (3, 3), activation='relu', padding = 'same', name='conv9c')(drop9b)
    drop9c = keras.layers.SpatialDropout2D(dropoutRate, name='drop9c')(conv9c)
    
    #10th Stage
    conv10 = keras.layers.Conv2D(2, (3, 3), activation='relu', padding = 'same', name='conv10')(drop9c)
    
    model = keras.models.Model(inputLayer, conv10)
    
#    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model