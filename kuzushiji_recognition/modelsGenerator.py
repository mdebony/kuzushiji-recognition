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
    model.fit(inputData[:nVal], outputData[:nVal], epochs=maxEpochs, batch_size=batchSize,
              validation_data=(inputData[nVal:nTest], outputData[nVal:nTest]), shuffle=True,
              callbacks=callbacks, verbose=1)

    model = keras.models.load_model(filenameModel)
    test_loss, test_acc = model.evaluate(inputData[nVal:nTest], outputData[nVal:nTest])

    return test_loss

def createKMNISTModel1(inputShape, outputLenght, dropoutRate, convLayer, denseLayer, kernelSize = (3,3), finalActivationFunction = 'softmax'):
    
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