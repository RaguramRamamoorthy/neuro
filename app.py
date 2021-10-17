from flask import Flask, render_template,request               #import
from keras.preprocessing import image
import numpy as np 
import os
from keras.models import load_model
from keras.models import Model



app = Flask(__name__)                                    #calling


def get_model():
    global model
    model = load_model('model.h5',compile=False)
    print("Model loaded!")
    return model



def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def prediction(img_path):
    print('-------------------------------')
    print(img_path)
    new_image = load_image(img_path)
    print(new_image)
    print('--------------------------------')
    model = get_model()

    pred = np.argmax(model.predict(new_image), axis=-1)
    
    print(pred)
    
    labels=np.array(pred)
    labels[labels>=0.6]=1
    labels[labels<0.6]=0
    
    print(labels)
    final=np.array(labels)
    
    if final[0][0]==1:
        return "No T"
    else:
        return "Good"

@app.route("/", methods=['GET', 'POST'])                 #initialising
def home():                                              #function call
    return render_template('home.html')                  #return and calling HTML page (designed template)


@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)



if __name__ == "__main__":
    app.run()                                            #run the application

