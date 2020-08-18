# save the final model to file
import sys
import os
from datetime import datetime
from matplotlib import pyplot
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

num_class = 1
model_num = 0


def get_num_cat():
    global num_class
    num_class = len(next(os.walk('Dataset/train'))[1])
    print("Number of cats in training dataset: " + str(num_class))
    cat_names = next(os.walk('Dataset/train'))[1]
    with open("models/classification/cat_names.txt", "w") as cat_name_file:
        for name in cat_names:
            cat_name_file.write("%s\n" % name)


def get_latest_model_num():
    global model_num
    files = os.listdir("models\classification")
    for file in files:
        if file.endswith(".h5"):
            model_num = int(str(file).split("_")[0])
    model_num += 1


def define_model():
    model = MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=(224, 224, 3))  # imports the mobilenet model.

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    # x = Flatten()(model.layers[-1].output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions.
    #x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    output = Dense(num_class, activation='sigmoid')(x)  # final layer with softmax activation
    model = Model(inputs=model.input, outputs=output)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, zoom_range=0.2, rotation_range=20,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    # prepare iterators
    train_generator = train_datagen.flow_from_directory('Dataset/train/',
                                                        target_size=(224, 224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode="categorical",
                                                        shuffle=True)
    test_generator = test_datagen.flow_from_directory('Dataset/val/',
                                                      target_size=(224, 224),
                                                      color_mode='rgb',
                                                      batch_size=32,
                                                      class_mode='categorical',
                                                      shuffle=True)

    history = model.fit(train_generator, steps_per_epoch=len(train_generator),
                        validation_data=test_generator, validation_steps=len(test_generator), epochs=10,
                        callbacks=my_callbacks)
    summarize_diagnostics(history)


if __name__ == "__main__":
    # entry point, run the test harness
    get_num_cat()
    get_latest_model_num()
    print('New model name: ' + str(model_num) + '_model-cat' + str() + '-e_{epoch:02d}-acc_{val_loss:.2f}.h5')
    my_callbacks = [
        EarlyStopping(patience=2),
        ModelCheckpoint(filepath='models/classification/' + str(model_num) + '_model_cats' + str(num_class) +
                                 '_e_{epoch:02d}_acc_{val_loss:.2f}.h5', save_best_only=True)
    ]
    run_test_harness()

