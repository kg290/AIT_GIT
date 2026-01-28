import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"kg-hackathon-e3f03b59d928.json"

from google.cloud import vision


from google.cloud import vision


client = vision.ImageAnnotatorClient()

with open("ce2ca0f38002e2f6e4392e8173cd2551.jpg", "rb") as f:
    content = f.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)

if response.text_annotations:
    print(response.text_annotations[0].description)
else:
    print("No text found")
