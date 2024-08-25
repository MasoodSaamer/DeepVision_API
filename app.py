#Handling api requests
from flask import Flask, request, jsonify #request handles the incoming data request. in our case image request. jsonify converts the python dictionary to JSON which is standard practice for web applications.
from flask_restful import Api, Resource #Api for creating REST API with Flask. Resource for API endpoint which basically handles HTTP Methods
import torch
from torchvision import transforms
from PIL import Image #PIL stands for Python Imaging Library from the Pillow package. Allows image manipulation in Python
import io #handles input/output operations. for example can convert a byte stream from an uploaded image into a format of PIL
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from flask_cors import CORS

app = Flask(__name__) #flask application instance. like the central point for the app
CORS(app, resources={r"/classify-image": {"origins": "*"}})  # This will enable CORS for all routes
api = Api(app) #creates api for the application(app)

# Define CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Set a custom cache directory. for AWS
torch.hub.set_dir('/tmp/.cache/torch/hub')

#now since we recieve image requests from user, we have to process them again for our model to work on it.
#also need to import our trained model which we trained in model.py
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('./models/cifar10_resnet18.pth', map_location=torch.device('cpu'))) #ensuring the data is loaded onto the cpu of the server we use. Can be switched to GPU if your server allows that
model.eval() #model to evaluation mode

transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

#Just some text to show the main route of the application is working
@app.route('/')
def index():
    return "Welcome to the Image Classification API by Saamer!"


#time to handle the image request and apply the trained model to it
class ImageClassification(Resource): #class which inherits from resource. will handle the logic for /classify-image route
    def post(self): #activated when a POST request is made to /classify-image.
        file = request.files['image'] #retrieving the file the user uploaded. request.files is a dictionary and image is the key
        img = Image.open(io.BytesIO(file.read())).convert('RGB') #image is first converted into byte stream using file.read() then its converted into a RGB format for PIL image object using Image.open()
        img = transform(img).unsqueeze(0) #applying the transformation defined earlier. unsqueeze(0) adds an extra dimension to tensor making it a batch size of 1 which is required by our model
        with torch.no_grad(): #dont need to compute the gradients. saves memory and speeds up
            outputs = model(img) #applies the model to the image. returns the predictions for each class of the model
            _, predicted = torch.max(outputs.data, 1) #returns the class with the highest prediction score with the specified dimension of 1
            class_name = class_names[predicted.item()]
        return jsonify({'class': class_name}) #.item gets the raw value from the PyTorch tensor. jsonify converts into JSOn so the server client can handle it
api.add_resource(ImageClassification, '/classify-image') #maps the ImageClassification resource to /classify-image endpoint

if __name__ == '__main__': #security to ensuring safe execution
    app.run(host='0.0.0.0', port=5000)
    


