from imports import *
from transformers import TFSegformerForSemanticSegmentation
from keras import backend
import numpy as np


mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])

def load_model():
    """ Model """
    model_checkpoint = "nvidia/mit-b1"
    id2label =  {0: "outer", 1: "landslide"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True   
    )
    model.load_weights("model-weights/segformer-5-b1.h5")
    return model

class Predict:

    def __init__(self, model):
        self.model = model

    def predict(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        input_image = tf.image.resize(image, (256, 256))
        input_image = tf.image.convert_image_dtype(input_image, tf.float32)
        input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
        input_image = tf.transpose(input_image, (2, 0, 1))
        input_image = tf.expand_dims(input_image, 0)
        pred = self.model.predict(input_image).logits
        mask = create_mask(pred).numpy()
        resized_img_array = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
        return resized_img_array

    

def pred_read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.0
    # image = image.astype(np.float32)
    return image

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]

def placeMaskOnImg(img, mask, color):
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def clear_dir(PATH):
    filenames = os.listdir(PATH)
    for filename in filenames:
        os.remove(os.path.join(PATH, filename))


if __name__ == '__main__':

    model = load_model()
    model.summary()
    output = predict('static/uploads/input.png')
    img = pred_read_image('static/uploads/input.png')
    color = np.array([158, 192, 247])/255.0
    pred = mask_parse(output)*255

    y = placeMaskOnImg(img, pred, color)
    plt.imsave(fname='output.png', arr=y)
    print("DOne")