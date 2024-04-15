import numpy as np
from classes import imagenet_classes
import grpc
from grpc import _channel

def run(image_path):
    # Create SSL/TLS channel credentials
    credentials = grpc.ssl_channel_credentials()

    # Create channel with HTTP/2 support and compression options
    options = [('grpc.default_compression_algorithm', grpc.Compression.Gzip)]
    channel = grpc.secure_channel('10.1.1.7:443', credentials, options)

    # Import the required gRPC modules
    from grpc.beta import implementations
    import tensorflow as tf
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

    # Create gRPC stub
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    with open(image_path, "rb") as f:
        img = f.read()

    # Create a request object
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.inputs['0'].CopyFrom(tf.make_tensor_proto(img, dtype=tf.string))

    # Make a gRPC call
    result = stub.Predict(request, 10.0)  # 10.0 is the timeout in seconds

    # Decode and print the result
    output = result.outputs['0'].float_val
    result_index = np.argmax(output)
    print(imagenet_classes[result_index])

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    run(image_path)
