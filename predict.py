import numpy as np
from classes import imagenet_classes
import grpc
from ovmsclient import make_grpc_client

def run(image_path):
    # Create SSL/TLS channel credentials
    credentials = grpc.ssl_channel_credentials()

    # Create channel with HTTP/2 support
    options = [('grpc.default_compression_algorithm', grpc.CompressionAlgorithm.Gzip)]
    channel = grpc.secure_channel('10.1.1.7:443', credentials, options)

    # Create gRPC client
    client = make_grpc_client(channel)

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
