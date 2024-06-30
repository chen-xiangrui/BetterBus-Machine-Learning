from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import easyocr
from gtts import gTTS
import pygame

# Define the path to the pretrained model and the source directory
model_path = Path('best.pt')

# Load pretrained YOLOv8s model
model = YOLO(model_path)

# translate text on image to string
def ocr_to_string(image):
    image_PIL_form = image
    image = np.array(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    result_list = [text_tuple[1] for text_tuple in result] # convert tuple to list
    result_string = ' '.join(result_list) # concat list string elements into a string
    list_of_bus_numbers = ['A1', 'A2', 'D1', 'D2', 'K', 'E', 'BTC', '96', '95', '151', 
                           'L', '153', '154', '156', '170', '186', '48', '67', '183', 
                           '188', '33', '10', '200', '201']
    
    # run detection model on bus number object
    model_path_cropped = Path('best-cropped.pt')
    model_cropped = YOLO(model_path_cropped)
    results_cropped = model_cropped(source=image_PIL_form, conf=0.40)
    image_cropped = None
    bus_number = 'Bus not found'
    for result_cropped in results_cropped:
        orig_img = result_cropped.orig_img
        
        for i, bbox in enumerate(result_cropped.boxes.xyxy):
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Crop and do image processing for the detected object
            image_cropped = Image.fromarray(orig_img).crop((xmin, ymin, xmax, ymax))
    
            # attempt 1 to identify bus number through bus number object
            image_cropped = np.array(image_cropped)
            reader_cropped = easyocr.Reader(['en'])
            result_cropped = reader_cropped.readtext(image_cropped)
            result_list_cropped = [text_tuple[1] for text_tuple in result_cropped] # convert tuple to list
            result_string_cropped = ' '.join(result_list_cropped) # concat list string elements into a string 
            bus_number = find_substring(result_string_cropped, list_of_bus_numbers)
            
        if bus_number != 'Bus not found':
            return bus_number
    
        # attempt 2 to identify bus number through bus object
        return find_substring(result_string, list_of_bus_numbers)
    
    # case where no bus object is detected (??)
    return 'Bus not found'

def find_substring(main_string, substrings):
    for substring in substrings:
        if substring in main_string:
            return substring
    return 'Bus not found'

def process_results(result):
    img = result.orig_img

    for i, bbox in enumerate(result.boxes.xyxy):  
        xmin, ymin, xmax, ymax = map(int, bbox)
        ymax -= (1 / 4) * (ymax - ymin)
        # Crop the detected object
        cropped_img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
        
        return ocr_to_string(cropped_img)
    
results = model(source=str('test/images'), conf=0.70)

for result in results:
    bus_result = process_results(result)
    print(Path(result.path).stem)
    print(bus_result)
    
    language = 'en'
    
    if bus_result != 'Bus not found' and bus_result is not None:
        try:
            # Extract the image name from the result object
            image_name = Path(result.path).stem  # Get the image name without extension
            audio_file_name = f'audio_{image_name}.mp3'  # Create the new audio file name
            
            audio_obj = gTTS(text='Bus number ' + bus_result + 'is approaching', lang=language, slow=False)
            audio_file = Path('audio') / audio_file_name  # Ensure audio file is saved in 'audio' directory
            audio_obj.save(audio_file)
            
            # Initialize the mixer module
            pygame.mixer.init()
            # Load the mp3 file
            pygame.mixer.music.load(str(audio_file))
            # Play the loaded mp3 file
            pygame.mixer.music.play()
            
            # Wait until the music finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    else:
        try:
            # Extract the image name from the result object
            image_name = Path(result.path).stem  # Get the image name without extension
            audio_file_name = f'audio_{image_name}.mp3'  # Create the new audio file name
            
            audio_obj = gTTS(text='Bus is too far away to detect', lang=language, slow=False)
            audio_file = Path('audio') / audio_file_name  # Ensure audio file is saved in 'audio' directory
            audio_obj.save(audio_file)
            
            
            # Initialize the mixer module
            pygame.mixer.init()
            # Load the mp3 file
            pygame.mixer.music.load(str(audio_file))
            # Play the loaded mp3 file
            pygame.mixer.music.play()
            
            # Wait until the music finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
