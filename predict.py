import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model/medical_model.h5")

classes = {
    0: "NORMAL",
    1: "PNEUMONIA",
    2: "TUMOR",
    3: "NO TUMOR"
}

def predict_disease(img_path):

    test_image = image.load_img(img_path, target_size=(128,128))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    test_image = test_image / 255.0

    result = model.predict(test_image)

    prediction = np.argmax(result)

    return classes[prediction]
