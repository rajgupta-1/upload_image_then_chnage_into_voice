from tkinter import filedialog, Tk
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from playsound import playsound

# Step 1: Select Image File
Tk().withdraw()  # Close the root window
file_path = filedialog.askopenfilename(title="Select Crop Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])

if not file_path:
    print(" No file selected.")
    exit()

# Step 2: Load and Display Image
image = Image.open(file_path).convert("RGB")
image.show()  # Opens in default image viewer

# Step 3: Load BLIP Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Step 4: Generate Caption
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)
print("📝 Image Caption:", caption)

# Step 5: Convert Caption to Voice
tts = gTTS(text="This image looks like " + caption, lang='en')
audio_file = "caption_audio.mp3"
tts.save(audio_file)
playsound(audio_file)  # Plays the mp3 file
