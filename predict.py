import numpy as np
from classes import imagenet_classes
import grpc

# Create SSL/TLS channel credentials
credentials = grpc.ssl_channel_credentials()

# Create channel with HTTP/2 support and compression options
options = [('grpc.default_compression_algorithm', grpc.Compression.Gzip)]
channel = grpc.secure_channel('10.1.1.7:443', credentials, options)

# Import the make_grpc_client function from ovmsclient
from ovmsclient import make_grpc_client

# Create gRPC client with the secure channel
client = make_grpc_client(channel)

with open("zebra.jpeg", "rb") as f:
    img = f.read()

# Make a gRPC call
output = client.predict({"0": img}, "resnet")

# Decode and print the result
result_index = np.argmax(output[0])
print(imagenet_classes[result_index])
