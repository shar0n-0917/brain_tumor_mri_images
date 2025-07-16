import jetson_inference
import jetson_utils
import argparse
import os # Import the os module for path manipulation

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the MRI image to process")
parser.add_argument("--network", type=str, default="resnet-18", help="model to use, can be: googlenet, resnet-18, ect. (see --help for others)")

# Construct absolute paths using os.path.expanduser to resolve '~'
# This is a safer way to handle user home directory paths in Python.
default_model_path = os.path.expanduser("~/jetson-inference/python/training/classification/models/brain-tumors/resnet18.onnx")
default_labels_path = os.path.expanduser("~/jetson-inference/python/training/classification/models/brain-tumors/labels.txt")

parser.add_argument("--model", type=str, default=default_model_path, help="path to custom model")
parser.add_argument("--labels", type=str, default=default_labels_path, help="path to class labels")

# --- MODIFICATION START ---
# Add arguments for custom input and output blob names
parser.add_argument("--input-blob", type=str, default="data", help="name of the input layer of the model (default: 'data')")
parser.add_argument("--output-blob", type=str, default="prob", help="name of the output layer of the model (default: 'prob')")
# --- MODIFICATION END ---

opt = parser.parse_args()

# Verify paths before passing to jetson_inference (optional, but good for debugging)
print(f"Attempting to load model from: {opt.model}")
print(f"Attempting to load labels from: {opt.labels}")

# --- MODIFICATION START ---
print(f"Using input blob name: '{opt.input_blob}'")
print(f"Using output blob name: '{opt.output_blob}'")
# --- MODIFICATION END ---

if not os.path.exists(opt.model):
    print(f"Error: Model file not found at {opt.model}")
    # Consider exiting here if the file is critical
    # import sys
    # sys.exit(1)
if not os.path.exists(opt.labels):
    print(f"Error: Labels file not found at {opt.labels}")
    # Consider exiting here if the file is critical
    # import sys
    # sys.exit(1)

img = jetson_utils.loadImage(opt.filename)

# --- MODIFICATION START ---
# Pass the input_blob and output_blob as keyword arguments
net = jetson_inference.imageNet(network=opt.network, model=opt.model, labels=opt.labels,
                                input_blob=opt.input_blob, output_blob=opt.output_blob)
# --- MODIFICATION END ---

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
print("MRI image shows "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")