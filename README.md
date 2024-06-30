# BetterBus Machine Learning
This repository allows you to be able to perform generation of bus number audios on a set of images.

## Training Process
### train/images
We store our images used for training in the train folder, where we upload the raw images and augmented images we collected from data augmentation.

### train.py
After storing the images, we run the train.py file, which loads the YOLOv8 model and performs training using the set of image datasets from train/ directory.

After training has completed, a weights file for bus object detection is provided (best.pt), which we can use for future bus detection purposes.

In our case, we loaded the best.pt weights file into our bus number detection model (detect.py).

## How to use our bus number detection model

### Installing the necessary dependencies required.

  `pip install ultralytics`
  
  `pip install Pillow`
  
  `pip install numpy`

  `pip install easyocr`

  `pip install gTTS`

  `pip install pygame`

### Inputting images for testing
To run your images for testing, move your images to the test/images directory.

Else, the current default repository provides you with 16 images in test/images already, for testing purposes.

### Running the bus number detection model
Navigate your way to detect.py file.

Run the file, or use the command:

`python detect.py`

### Results
The audio of the bus number will be saved into the audio folder, with the respective image's audio under the name: audio_{file_name}.mp3

Concurrently, the audio will also be played out while the detect.py file is running.

Do note that the audio of the bus number would be played if the bus number can be extracted. Else, the audio will play 'Bus is too far away to detect'.
