# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout,Average,Subtract,Multiply,Lambda
from keras.optimizers import Adam,SGD
from keras.utils import plot_model
from keras import backend as K           

def unet_model(n_classes=4, im_sz=32, n_channels=3, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.2, 0.5, 0.1]):
    droprate=0.25
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    inputs2 = Input((im_sz, im_sz, n_channels))
    inputs2n = Input((im_sz, im_sz, n_channels))
    inputs2f = Input((im_sz, im_sz, n_channels))

    #inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_1 = Dropout(droprate)(pool4_1)

    n_filters *= growth_factor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_2 = Dropout(droprate)(pool4_2)

    n_filters *= growth_factor
    convt5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
    convt5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convt5)
    

    #net2
    n_filters2 = n_filters_start
    #inputs = BatchNormalization()(inputs)
    convs1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs2)
    convs1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convs1)
    pools1 = MaxPooling2D(pool_size=(2, 2))(convs1)
    #pool1 = Dropout(droprate)(pool1)

    n_filters2 *= growth_factor
    pools1 = BatchNormalization()(pools1)
    convs2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pools1)
    convs2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convs2)
    pools2 = MaxPooling2D(pool_size=(2, 2))(convs2)
    pools2 = Dropout(droprate)(pools2)

    n_filters2 *= growth_factor
    pools2 = BatchNormalization()(pools2)
    convs3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pools2)
    convs3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convs3)
    pools3 = MaxPooling2D(pool_size=(2, 2))(convs3)
    pools3 = Dropout(droprate)(pools3)

    n_filters2 *= growth_factor
    pools3 = BatchNormalization()(pools3)
    convs4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pools3)
    convs4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convs4_0)
    pools4_1 = MaxPooling2D(pool_size=(2, 2))(convs4_0)
    pools4_1 = Dropout(droprate)(pools4_1)

    n_filters2 *= growth_factor
    pools4_1 = BatchNormalization()(pools4_1)
    convs4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pools4_1)
    convs4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convs4_1)
    pools4_2 = MaxPooling2D(pool_size=(2, 2))(convs4_1)
    pools4_2 = Dropout(droprate)(pools4_2)

    n_filters2 *= growth_factor
    convs5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pools4_2)
    convs5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(convs5)
    


    #submodel
    submodel=Model(inputs=inputs, outputs=convt5)

    #convs52=submodel(inputs2)
    conv5n=submodel(inputs2n)
    conv5f=submodel(inputs2f)
    print(conv5n.shape)
    print(conv5f.shape)   
    dif_near=Subtract()([conv5n, convt5])
    dif_far=Subtract()([conv5f, convt5])
    
    
    far=Lambda(lambda dif_far: K.abs(1/(1+dif_far)))(dif_far)
    near=Lambda(lambda dif_near: K.abs(dif_near))(dif_near)   

     
    mul=Multiply()([far,near]) 

    #conpare feature
    conv5 = Average()([convt5, convs5])
    subs=Subtract()([convt5, convs5])

    n_filters //= growth_factor
    if upconv:
        up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
    else:
        up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
    conv6_1 = Dropout(droprate)(conv6_1)

    n_filters //= growth_factor
    if upconv:
        up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
    conv6_2 = Dropout(droprate)(conv6_2)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(droprate)(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(droprate)(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs,inputs2,inputs2n,inputs2f], outputs=[conv10,subs,mul])

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=[weighted_binary_crossentropy,'MSE','MSE'],loss_weights=[1, 0.8,0.1])
    return model
