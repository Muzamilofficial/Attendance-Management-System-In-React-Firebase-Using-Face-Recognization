import cv2
import face_recognition
import os
import pickle

# Load images from local folder
folderPath = "backend/output_images"
pathList = os.listdir(folderPath)
imgList = []
stdList = []
missing_face_images = []  # List to store names of images with no faces detected

print("Images to process:", pathList)

for path in pathList:
    imgstd = cv2.imread(os.path.join(folderPath, path))
    if imgstd is None:
        print(f"Warning: Unable to load image {path}")
        continue
    imgstd = cv2.resize(imgstd, (1400, 1650))  # Resize image
    imgList.append(imgstd)
    stdList.append(os.path.splitext(path)[0])

# Function to generate face encodings and track missing face images
def findEncodings(imagesList, imageNames):
    encodeList = []
    missing_faces = []  # List to store names of images with no faces detected

    for img, name in zip(imagesList, imageNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)  # Detect face locations
        if len(face_locations) > 0:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        else:
            print(f"Warning: No face detected in image {name}.")
            missing_faces.append(name)  # Add image name to missing_faces list

    return encodeList, missing_faces

# Generate encodings and track missing face images
print("Encoding Started......")
encodeListKnown, missing_face_images = findEncodings(imgList, stdList)

if len(encodeListKnown) > 0:
    encodeListKnownWithIDs = [encodeListKnown, stdList]
    print("Encoding Complete")

    file = open("Encode.p", "wb")
    pickle.dump(encodeListKnownWithIDs, file)
    file.close()
    print("File Saved")
else:
    print("No encodings generated. Check if faces are present in the images.")

# Print names of images with no faces detected
if missing_face_images:
    print("Images with no faces detected:")
    for image_name in missing_face_images:
        print(image_name)
else:
    print("All images had detectable faces.")