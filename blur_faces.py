import cv2
import matplotlib.pyplot as plt

# Function to blur faces
def blur_faces(image, face_coordinates):
    for (x, y, w, h) in face_coordinates:
        # Extract the region of interest (ROI) where the face is located
        roi = image[y:y+h, x:x+w]
        
        # Apply Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(roi, (99, 99), 30)
        
        # Replace the original face with the blurred face
        image[y:y+h, x:x+w] = blurred_face

    return image

# Read the image
imagePath = 'people.jpg'
img = cv2.imread(imagePath)

# Convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Detect faces using the classifier
faces = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Blur the faces
img_with_blurred_faces = blur_faces(img, faces)

# Convert BGR to RGB for displaying using plt
img_rgb = cv2.cvtColor(img_with_blurred_faces, cv2.COLOR_BGR2RGB)

# Display the image with blurred faces
plt.imshow(img_rgb)
plt.axis('off')
plt.imsave('people_blurred.jpg', img_rgb)
plt.show()