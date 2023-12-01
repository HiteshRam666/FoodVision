
### 1. Imports and class names setup###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open('class_names.txt', 'r') as f:
  class_names = [food_name.strip() for food_name in f.readlines()]

### 2. Model and transforms preparation###
# Create model and transforms
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes = 101)

# Load saved weights 
effnetb2.load_state_dict(
  torch.load(f = 'effnetb2_Food101.pth',
              map_location = torch.device('cpu'))
)

### 3. Predict Function ###
def predict(img) -> Tuple[Dict, float]:
  # Start a timer 
  start_time = timer()

  # Transform the input image for use with effnetb2
  img = effnet_b2_transforms(img).unsqueeze(0) 

  # Put model into eval mode, make prediction
  effnetb2.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probabilities
    pred_probs = torch.softmax(effnetb2(img), dim = 1)

  # Create a prediction label and prediction probability dict
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time

### 4. Gradio app ###
# Create title, description and article
title = 'FoodVision 🍔🍕🍰👁'
description = 'An EfficientNetB2 Feature extractor'
article = 'Pytorch model deployment'

# Create example list 
example_list = [['1.jpg'],
                ['2.jpg'],
                ['3.jpg'],
                ['4.jpg'],
                ['5.jpg']]

# Create gradio demo
demo = gr.Interface(fn = predict,
                    inputs = gr.Image(type = 'pil'),
                    outputs = [gr.Label(num_top_classes = 5, label = 'Predictions'),
                               gr.Number(label = 'Prediction time (s)')],
                    examples = example_list,
                    title = title,
                    description = description,
                    article = article)

demo.launch(debug = False)
