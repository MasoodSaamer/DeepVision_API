# Image Classification API by Saamer

## Table of Contents
- [Project Overview](#project-overview)
- [About API](#About-Api)
- [Usage](#usage)
- [Testing](#testing)
- [Security and SSL](#security-and-ssl)
- [API Endpoints](#api-endpoints)
- [License](#license)

## Project Overview
This project is an image classification API built using Flask in Python. It uses a pre-trained ResNet18 model on the CIFAR-10 dataset to classify images into one of 10 categories offered by CIFAR-10. The API is containerized using Docker and deployed on AWS using an EC2 instance, with the Docker image pulled from Amazon Elastic Container Registry (ECR). For security, the API is served via NGINX with SSL certificates provided by Cloudflare.

## About API
This API provides image classification capabilities using a deep learning model. The model is based on ResNet18, a popular convolutional neural network (CNN) architecture, and is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes. 
The API allows users to upload an image and receive a classification result. It is designed to be easily deployable in a Docker container and is hosted on an AWS EC2 instance for scalability and reliability.

## Usage

### API Implementation in a Web Application

For using the API, you can visit www.imageaibysaamer.com 

## Testing

To test the API locally or after deployment, follow these steps:

### 1. Build and Run the Docker Container

1. **Build the Docker Image:**
   - Run the following command to build the Docker image:
     ```bash
     docker build -t your-image-name .
     ```

2. **Run the Docker Container:**
   - Once the image is built, you can run the container using:
     ```bash
     docker run -d -p 5000:5000 your-image-name
     ```
   - This will start the API, making it accessible on `http://localhost:5000`.

### 2. Testing the API locally with Postman

1. **Open Postman.**

2. **Create a new POST request:**
   - Set the request type to `POST`.
   - Enter the URL: `http://localhost:5000/classify-image` 

3. **Add the image file:**
   - Under the `Body` tab, select `form-data`.
   - Add a new key with the name `image`.
   - Set the type to `File`.
   - Choose the image file you want to upload for classification.

4. **Send the request:**
   - Click the `Send` button to submit the request to the API.

5. **View the response:**
   - The response will include the classification result in JSON format. For example:
     ```json
     {
       "class": "automobile"
     }
     ```
   - The `class` field will indicate the predicted category for the uploaded image.

## Security and SSL

### NGINX Configuration

To ensure secure access, the API is served through NGINX, which handles SSL termination. In my case, the SSL certificates are provided by Cloudflare, ensuring encrypted communication between the client and the server.

### SSL Certificates

- **Cloudflare SSL:** SSL certificates are generated and managed through Cloudflare.
- **NGINX Configuration:** NGINX is configured to use these certificates to serve the Flask application over HTTPS.

## API Endpoints

### `POST /classify-image`
- **Description:** Uploads an image and returns the classification result.
- **Parameters:** 
  - `image` (file): The image file to be classified.
- **Response:**
  - `200 OK`: JSON response containing the predicted class.

### `GET /health`
- **Description:** Checks if the API is running.
- **Response:**
  - `200 OK`: Returns "API is up and running."


## License

This project is licensed under the Creative Commons Attribution 4.0 International License. You are free to use, modify, and distribute this software, provided that you give appropriate credit to the original author. See the [LICENSE](LICENSE) file for more details.


