from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def read_image(pic_url):
    response = requests.get(pic_url)
    img = Image.open(BytesIO(response.content))
    img.save("img1.png", "PNG")
    img = cv2.imread("img1.png", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    return img


def prep_data(pic_url):
    image = read_image(pic_url)
    data = np.ndarray((1, 3, 64, 64), dtype=np.uint8)
    data[0] = image.T
    return data


def dog_or_cat(pic_url):
    model = load_model("DogCatmodel")
    images = prep_data(pic_url)
    predictions = model.predict(images)
    for i in range(0, len(predictions)):
        if predictions[i, 0] >= 0.5:
            answer = 'I am {:.2%} sure this is a Dog'.format(predictions[i][0])
        else:
            answer = 'I am {:.2%} sure this is a Cat'.format(1 - predictions[i][0])
    return answer

# dog_or_cat('https://static.scientificamerican.com/sciam/cache/file/92E141F8-36E4-4331-BB2EE42AC8674DD3_source.jpg?w=590&h=800&62C6A28D-D2CA-4635-AA7017C94E6DDB72')
