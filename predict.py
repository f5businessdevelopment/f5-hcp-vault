import numpy as np
from classes import imagenet_classes
import grpc
from ovmsclient import make_grpc_client

def run(image_path):
    # Create SSL/TLS channel credentials
    credentials = grpc.ssl_channel_credentials()

    # Replace '10.1.1.7:443' with the address of your gRPC server and the SSL port
    # You may need to replace '443' with the actual SSL port of your server
    with grpc.secure_channel('10.1.1.7:443', credentials) as channel:
        client = make_grpc_client('10.1.1.7:443', tls=True)  # Pass the URL as a string here

        with open(image_path, "rb") as f:
            img = f.read()

        # Make a gRPC call
        output = client.predict({"0": img}, "resnet")

    # Decode and print the result
    result_index = np.argmax(output[0])
    print(imagenet_classes[result_index])

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    run(image_path)
