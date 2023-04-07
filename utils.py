from imports import *

def focal_dice_loss(y_true, y_pred, gamma=2.0, alpha=0.25):

    dice_loss = sm.losses.DiceLoss()(y_true, y_pred)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = tf.keras.backend.pow(1.0 - pt, gamma) * tf.keras.backend.binary_crossentropy(y_true, y_pred)

    return (alpha * focal_loss) + ((1 - alpha) * dice_loss)

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    """ Addition """
    x = x + s
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs) 
    x = Concatenate()([x, skip_features]) 
    x = residual_block(x, num_filters, strides=1) 
    return x

def build_resunet(input_shape):
    """ RESUNET Architecture """

    inputs = Input(input_shape)

    """ Endoder 1 """ 
    x = Conv2D(64, 3, padding="same", strides=1)(inputs) 
    x = batchnorm_relu(x) 
    x = Conv2D(64, 3, padding="same", strides=1)(x) 
    s = Conv2D(64, 1, padding="same")(inputs) 
    s1 = x + s

    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)

    """ Bridge """
    b = residual_block(s3, 512, strides=2)

    """ Decoder 1, 2, 3 """ 
    x = decoder_block(b, s3, 256) 
    x = decoder_block(x, s2, 128) 
    x = decoder_block(x, s1, 64)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, outputs, name="RESUNET")

    return model

def load_model():
    """ Model """
    model = build_resunet((128, 128, 3))
    model.compile(optimizer = tf.keras.optimizers.Adam(1e-3), 
              loss = focal_dice_loss, 
              metrics = sm.metrics.iou_score)
    
    model.load_weights('model-weights/resunet_20_augmented_focalDiceLoss.h5')
    return model

def pred_read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128,128))
    image = image/255.0
    return image

def pred_read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def placeMaskOnImg(img, mask, color):
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img

def clear_dir(PATH):
    filenames = os.listdir(PATH)
    for filename in filenames:
        os.remove(os.path.join(PATH, filename))

if __name__ == '__main__':

    model = load_model()
    model.summary()

    img = pred_read_image('image_52.png')
    pred = model.predict(np.expand_dims(img, axis=0))[0] > 0.5
    pred = mask_parse(pred)*255

    color = np.array([158, 192, 247])/255.0
    output = placeMaskOnImg(img, pred, color)
    
    plt.imsave(fname="output.png", arr=output)
