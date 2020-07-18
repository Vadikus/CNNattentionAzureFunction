from datetime import datetime
import logging
import os

from urllib.request import urlopen
from PIL import Image
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from urllib.request import urlopen

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import time
import io
import json, base64

tf.compat.v1.disable_eager_execution()

def _log_msg(msg):
    logging.info("{}: {}".format(datetime.now(),msg))

model = None

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)
filename = os.path.join(scriptdir, 'mobilenetv2.h5')

heatmap_scema = [
    [0.0, (0, 0, 0)],
    #[0.20, (0, 0, 0)],
    #[0.30, (0, 0, 0.3)],
    #[0.50, (0, 0.5, 0)],
    #[0.70, (0, .7, 0.2)],
    [1.00, (0, 1.0, 0)],
]


def pixel(x, width=256, map=[], spread=1):
    width = float(width)
    r = sum([gaussian(x, p[1][0], p[0] * width, width/(spread*len(map))) for p in map])
    g = sum([gaussian(x, p[1][1], p[0] * width, width/(spread*len(map))) for p in map])
    b = sum([gaussian(x, p[1][2], p[0] * width, width/(spread*len(map))) for p in map])
    return min(1.0, r), min(1.0, g), min(1.0, b)


def gaussian(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d


def _initialize():
    global model

    start_time = time.time()
    if not model:
        #if os.environ.get('HOME') == '/home':
        #    model = load_model('/keras/vgg16.h5')
        #else:
        #    model = load_model('vgg16.h5')
        model = load_model(filename)

    logging.warning(f"Initializing weights: @ {time.time() - start_time :.2f} sec")
    return f"Initializing weights: @ {time.time() - start_time :.2f} sec"

def _predict_image(img_org):
    start_time = time.time()
    result = {}
    response = "Computer Vision challenge\r\n#AIApril\r\n\r\nTop 3 predictions of MobileNet_V2 \r\ntrained on ImageNet dataset:\r\n"
    img = img_org.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    for i in range(3):
        res = decode_predictions(preds, top=3)[0][i]
        response += f"{res[1]} - {res[2]*100:.2f}% \r\n"
        logging.info(f"Prediction: {res[1]} - {res[2]*100:.1f}%")
        result["prediction"+str(i)] = f"{res[1]} - {res[2]*100:.1f}%"

    #response += "\r\nVisualizing CAM (Class Activation Mapping)\r\naka Neural Network attention for the best prediction"

    #response += f"\r\nImage load + Predictions: @ {time.time() - start_time :.2f} sec \r\n"
    logging.info(f"\r\nImage load + Predictions: @ {time.time() - start_time :.2f} sec")

    ind = np.argmax(preds[0])

    vector = model.output[:, ind]

# 4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_15_project (Conv2D)       (None, 7, 7, 160)    153600      block_15_depthwise_relu[0][0]    
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_15_project_BN (BatchNorma (None, 7, 7, 160)    640         block_15_project[0][0]           
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_15_add (Add)              (None, 7, 7, 160)    0           block_14_add[0][0]               
# [4/1/2020 5:02:21 AM]                                                                  block_15_project_BN[0][0]        
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_expand (Conv2D)        (None, 7, 7, 960)    153600      block_15_add[0][0]               
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_expand_BN (BatchNormal (None, 7, 7, 960)    3840        block_16_expand[0][0]            
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_expand_relu (ReLU)     (None, 7, 7, 960)    0           block_16_expand_BN[0][0]         
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_depthwise (DepthwiseCo (None, 7, 7, 960)    8640        block_16_expand_relu[0][0]       
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_depthwise_BN (BatchNor (None, 7, 7, 960)    3840        block_16_depthwise[0][0]         
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_depthwise_relu (ReLU)  (None, 7, 7, 960)    0           block_16_depthwise_BN[0][0]      
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_project (Conv2D)       (None, 7, 7, 320)    307200      block_16_depthwise_relu[0][0]    
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] block_16_project_BN (BatchNorma (None, 7, 7, 320)    1280        block_16_project[0][0]           
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] Conv_1 (Conv2D)                 (None, 7, 7, 1280)   409600      block_16_project_BN[0][0]        
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] Conv_1_bn (BatchNormalization)  (None, 7, 7, 1280)   5120        Conv_1[0][0]                     
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] out_relu (ReLU)                 (None, 7, 7, 1280)   0           Conv_1_bn[0][0]                  
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] global_average_pooling2d (Globa (None, 1280)         0           out_relu[0][0]                   
# [4/1/2020 5:02:21 AM] __________________________________________________________________________________________________
# [4/1/2020 5:02:21 AM] Logits (Dense)                  (None, 1000)         1281000     global_average_pooling2d[0][0]   
# [4/1/2020 5:02:21 AM] ==================================================================================================    # 
    # The output feature map of the last convolutional layer
    last_conv_layer = model.get_layer('Conv_1_bn')

    # The gradient of the vector class with regard to the output feature map
    grads = K.gradients(vector, last_conv_layer.output)[0]

    # A vector of shape (1280,), where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of conv layer, given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays, given the image
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array by "how important this channel is" with regard to the predicted class
    for i in range(1280):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    #response += f"Activation layers: @ {time.time() - start_time :.2f} sec \r\n"
    logging.info(f"Activation layers: @ {time.time() - start_time :.2f} sec")

    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    RGBheat = []
    for line in heatmap:
        RGBheat.append([])
        for x in line:
            r, g, b = pixel(x, width=1, map=heatmap_scema)
            r, g, b = [int(256*v) for v in (r, g, b)]
            pix = (r, g, b)
            RGBheat[-1].append(pix)

    heatmap = np.array(RGBheat)
    heatmap = np.uint8(heatmap)
    #heatmap = np.expand_dims(heatmap, axis=0)
    
    #response += f"HeatMap created: @ {time.time() - start_time :.2f} sec \r\n"
    logging.info(f"HeatMap created: @ {time.time() - start_time :.2f} sec")

    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize( img_org.size)

    heatmap = np.uint8(heatmap)

    superimposed_img = heatmap * 0.8 + img_org
    ## superimposed_img = img_org.copy()
    ## superimposed_img.putalpha(heatmap)

    ## result_img = Image.new("RGB", superimposed_img.size, (255, 255, 255))
    ## result_img.paste(superimposed_img, mask=superimposed_img.split()[3]) #
    result_img = image.array_to_img(superimposed_img)

    draw = ImageDraw.Draw(result_img)
    font = ImageFont.load_default()
    
    #response += f"\r\nTotal execution time: {time.time() - start_time :.2f} sec\r\n"
    logging.info(f"\r\nTotal execution time: {time.time() - start_time :.2f} sec")
    
    draw.text( (10,10), response, (55, 25, 255), font=font)

    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='JPEG', quality=100)

    result['img'] = base64.encodebytes(img_byte_arr.getvalue()).decode("utf-8")

    #result_img.save('test.jpg')

    return result     

def predict_image_from_url(image_url):
    start_time = time.time()
    logging.info("Predicting from url: " + str(image_url))
    ttt = _initialize()
    if image_url:
        with urlopen(image_url) as testImage:
            image_org = Image.open(testImage)
            #image_org = image.load_img(testImage)
            logging.info(f"\r\nTotal execution time: {time.time() - start_time :.2f} sec")
            return _predict_image(image_org)
    return
