from model import resnet50
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf

im_height = 224
im_width = 224

# Load the image that need to be classified
img = Image.open("test.jpg")
# resize 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# Preprocessing the loaded image
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
img = np.array(img).astype(np.float32)
img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
img = (np.expand_dims(img, 0))

# Save the dictionary to be used into class_indices.json
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# Adjust the pretrained model to output desired number of predictions
feature = resnet50(num_classes=2, include_top=False)
feature.trainable = False
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(2),
                             tf.keras.layers.Softmax()])

# Load the weigths of pretrained model
model.load_weights('./save_weights/resNet_101.ckpt')
# Predict the image style using pretrained model
result = model.predict(img)
prediction = np.squeeze(result)
predict_class = np.argmax(result)
# Print the classification results
print('The predicted image style belongs to：', class_indict[str(predict_class)], ' The level of confidence is：', prediction[predict_class])
plt.show()
