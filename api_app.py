import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
api = Api(app)

'''
if not os.path.isfile('iris-model.model'):
    train_model()

model = joblib.load('iris-model.model')
'''

classes = ['Action', 'Adventure', 'Animation', 'Biography',
		  'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
		  'History', 'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News',
		  'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War',
		  'Western']


class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        file = request.files['file']
        file.save('temp.jpeg')
        path = ''
        # load json and create model
        json_file = open(path + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(path + 'model.h5')
        print("Loaded model from disk")

        img = image.load_img(path +'temp.jpeg',target_size=(250,250,3))
        img = image.img_to_array(img)
        img = img/255

        proba = model.predict(img.reshape(1,250,250,3))

        top = np.argsort(proba[0])[:-2:-1]
        predicted_class = classes[top[0]]
        conf = format(proba[0][top[0]], '.3f')
        return jsonify({
            'Predicted_Class': predicted_class,
            'Confidence' : conf
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)