from ultralytics import YOLO
from pathlib import Path
import yaml

# Define paths
model_path = Path('betterbus-ml/best.pt')
data_yaml_path = Path('data.yaml')

# Load the data.yaml file
with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

# Get the validation images path from data.yaml
val_images_path = Path(data_yaml['val'])

# Check if the trained model path exists
if model_path.exists():
    # Load model
    model = YOLO(model_path)

    # Perform validation
    results = model(val_images_path, imgsz=640, task='val', conf=0.7)
    
    # Define save directory
    save_dir = Path('betterbus-ml/val/predict')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    for result in results:
        save_path = save_dir / Path(result.path).name
        result.save(save_path)

    # Print results (optional)
    print(results)
else:
    print(f"Model weights not found at {model_path}. Please ensure the training process completed successfully.")
