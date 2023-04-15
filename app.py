from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

dic={0 :'Unhealthy - Tomato Bacterial spot',
 1 :'Unhealthy - Tomato Early blight',
 2 :'Unhealthy - Tomato Late blight',
 3 :'Unhealthy - Tomato Leaf Mold',
 4 :'Unhealthy - Tomato Septoria leaf spot',
 5 :'Unhealthy - Tomato Spider mites Two-spotted spider mite',
 6 :'Unhealthy - Tomato Target Spot',
 7 :'Unhealthy - Tomato Yellow Leaf Curl Virus',
 8 :'Unhealthy - Tomato mosaic virus',
 9 :'Healthy - Tomato',
 10 :'Unhealthy - Cauliflower alternia leaf spot',
 11 :'Unhealthy - Cauliflower aphid colony',
 12 :'Unhealthy - Cauliflower black leg',
 13 :'Unhealthy - Cauliflower bugs attack',
 14 :'Unhealthy - Cauliflower downy mildew',
 15 :'Healthy',
 16 :'Unhealthy - mango',
 17 :'Healthy - mango',
 18 :'Unhealthy - Cauliflower ring spot',
 19 :'Unhealthy - Cauliflower white rust'}

model = load_model('path_to_my_model.h5')

model.make_predict_function()


def predict_label(img_path):
	i = image.load_img(img_path, target_size=(150,150))
	i = image.img_to_array(i)/255.0
	i = np.expand_dims(i,axis=0)
	i=np.vstack([i])
	p = model.predict_classes(i)
	return dic[p[0]] 

'''
i=0
def predict_label(img_path):
   
    img=image.load_img(img_path,target_size=(150,150))
    #plt.imshow(img)
    #plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    pred=model.predict(images,batch_size=1) 
    if pred[0][0]>0.6:
        category="Healthy"
    elif pred[0][1]>0.5:
        category="Unhealthy"
    print(category)
    return dic[pred[0]] 
'''
@app.route("/")
def hello():
    return  render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True,port=12345)