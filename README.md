# AI Classification Model for Brain Tumor Detection 🧠

This project details an **AI classification model** powered by the **Nvidia Jetson Orin Nano** machine. Its primary function is to identify the presence of brain tumors and classify them into specific types: **meningioma**, **glioma**, or **pituitary tumors**.

The model aims to:

  * **Aid doctors** in making more accurate diagnoses and providing faster prognoses.
  * **Educate students** (medical or general) about brain tumors and their visual characteristics in MRI images.

**Note:** This tool is primarily **educational**. For real-life medical situations, it is strongly advised to prioritize the judgment of **medical professionals** before using this tool as a secondary reinforcement.

-----

## The Algorithm ⚙️

The development and deployment of this AI model follow these steps:

1.  **Find a dataset:** Locate a dataset on Kaggle with a minimum size of 100 MB in `.csv` format.
2.  **Download dataset:** Download the chosen dataset from the site.
3.  **Extract files:** Use the `tar` function to extract the downloaded files.
4.  **Run Docker container:** Execute a Docker-closed container to run the necessary code.
5.  **Run training script:** Inside the Docker container, run the training script:
    ```bash
    python3 train.py --model-dir=models/brain-tumors data/brain-tumors
    ```
    (Note: `train.py` is a specific Python command that initiates training.)
6.  **Export the model:** Export the trained model so it can be used outside the Docker environment:
    ```bash
    python3 onnx_export.py --model-dir=models/brain-tumors
    ```
7.  **Locate the model:** The exported model will be found under `resnet18.onnx`.
8.  **Write Python code with argparse:** Develop Python code using `argparse` to guide the model to the files it needs to classify and the appropriate network. This code is run in the terminal.
9.  **Final output:** The model should be capable of assessing any given brain MRI image and printing out its classification.

-----

## Python Code 🐍

```python
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


# --- ADD THIS TO OVERLAY AND SAVE THE IMAGE ---

# Create the overlay text
overlay_text = f"{class_desc} ({confidence * 100:.2f}%)"

# Draw the overlay text on the image (top-left corner at x=10, y=10)
font = jetson_utils.cudaFont()

#font.OverlayText(img, 10, 10, overlay_text, 32, (255, 255, 255, 255), (0, 0, 0, 160))
font.OverlayText(img, img.width, img.height, overlay_text, 10,10, font.Red)

# Define output filename (can customize or generate dynamically)
output_filename = "classified_output.jpg"

# Save the image with the overlay
jetson_utils.saveImage(output_filename, img)
print(f"Image saved with overlay as: {output_filename}")
```

-----

## Explanation of the Python Code 📝

The Python script utilizes several key libraries to perform its image classification task:

  * **`jetson_inference`**: This is the **core library** for conducting inference operations on NVIDIA Jetson devices. It provides functionalities for loading machine learning models and classifying images.
  * **`jetson_utils`**: A **utility library** that complements `jetson_inference`. It offers functions for loading and saving images, as well as image manipulation (e.g., drawing text overlays).
  * **`argparse`**: This module is used for **parsing command-line arguments**. It enables users to customize the script's behavior by specifying inputs like the image file, model, and other parameters directly from the command line.
  * **`os`**: Provides an interface for **interacting with the operating system**. It's particularly useful for path manipulation, such as resolving user home directories (`~`) and checking if files exist.

### Script Flow:

1.  **Argument Parsing**: The script begins by setting up an **argument parser** using `argparse`. This allows users to provide various inputs when running the script from the command line:

      * `filename` (positional argument): **Required**. Specifies the path to the MRI image file to be processed.
      * `--network` (optional): Determines the **pre-trained model** to use if a custom model isn't provided. The default is `"resnet-18"`. Other options like `"googlenet"` are also available.
      * `--model` (optional): Specifies the **absolute path** to a custom ONNX model file. The default path is set to `~/jetson-inference/python/training/classification/models/brain-tumors/resnet18.onnx`. The `os.path.expanduser('~')` function correctly resolves the user's home directory.
      * `--labels` (optional): Specifies the **absolute path** to a text file containing the class labels corresponding to the custom model. The default path is `~/jetson-inference/python/training/classification/models/brain-tumors/labels.txt`.
      * `--input-blob` (optional, **MODIFIED**): Specifies the **name of the input layer** (blob) of the neural network model. The default is `"data"`. This is crucial for models that might have different input layer names.
      * `--output-blob` (optional, **MODIFIED**): Specifies the **name of the output layer** (blob) of the neural network model. The default is `"prob"`. This is important for models with varying output layer names.
        After defining the arguments, `parser.parse_args()` processes the command-line inputs and stores them in the `opt` object.

2.  **Path Verification**: The script then prints the resolved paths for the model and labels files, along with the specified input/output blob names. It also includes **basic checks** using `os.path.exists()` to verify if the model and labels files actually exist at the specified paths. If a file is not found, an error message is printed. (The commented-out `sys.exit(1)` lines show where the script could be configured to exit if these files are critical).

3.  **Image Loading**: The `jetson_utils.loadImage()` function is used to load the specified MRI image (`opt.filename`) into a `cudaImage` object. This is a **GPU-accelerated image format** suitable for processing with `jetson-inference`. This step basically loads the image with the label for display.

4.  **Image Classification**: The loaded `cudaImage` (`img`) is passed to the `net.Classify()` method. This performs the **actual inference**, classifying the image. It returns:

      * `class_idx`: The **index** of the predicted class (e.g., 0 for "no tumor", 1 for "tumor").
      * `confidence`: The **confidence score** (a float between 0 and 1) that the image belongs to the predicted class.

5.  **Result Output**: The `net.GetClassDesc()` method retrieves the **human-readable description** (label) for the predicted `class_idx`. The script then prints the classification result, including the class description, its index, and the confidence level as a percentage.

6.  **Image Overlay and Saving**:

      * `overlay_text`: A string is formatted to display the predicted class description and its confidence percentage (formatted to two decimal places).
      * `jetson_utils.cudaFont()`: An instance of `cudaFont` is created, which is used for drawing text on `cudaImage` objects.
      * `font.OverlayText()`: This method draws the `overlay_text` onto the `img`.
          * The first argument is the `cudaImage` to draw on.
          * `img.width` and `img.height` are used for positioning, though the `10,10` following it suggests an attempt to place it at (10,10) from the top-left. There might be a slight confusion in the arguments here, as `OverlayText` typically takes x, y coordinates directly. The `font.Red` specifies the color of the text.
          * The `font.OverlayText` method has been modified from the default `(255, 255, 255, 255)` (white) and `(0, 0, 0, 160)` (semi-transparent black background) to `font.Red` for the text color. The `32` (font size) and `10, 10` (x, y coordinates) are also adjusted.
      * `output_filename`: A string variable defines the name for the output image file.
      * `jetson_utils.saveImage()`: This function saves the modified `cudaImage` (which now includes the overlay text) to the specified `output_filename`. A confirmation message is printed to the console.

-----

## Running This Project ▶️

To execute this project, follow these commands in your terminal:

1.  Navigate to the project directory:
    ```bash
    cd brain_tumor_mri_images
    ```
2.  Run the Python script with an image for classification. You can either download an image you want to classify or use an image already present within the `test-images` directory:
    ```bash
    python3 finalproject.py test-images/Te-gl_0021.jpg --input-blob=input_0 --output-blob=output_0
    ```

-----

For more information, you can refer to the following YouTube link: [https://youtu.be/Uoy1po8qWwQ ](https://www.google.com/search?q=https://youtu.be/Uoy1po8qWwQ%C2%A0)