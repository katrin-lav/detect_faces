import sys
import os
import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace

def main(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    # Detect faces
    faces = RetinaFace.detect_faces(image_path)
    ages = []
    genders = []
    
    if not faces:
        print("No faces detected.")
        return



    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for key in faces.keys():
        x1, y1, x2, y2 = faces[key]['facial_area']
        face_crop = img_rgb[y1:y2, x1:x2]

        try:
            analysis = DeepFace.analyze(face_crop, actions=['age', 'gender'], enforce_detection=False, detector_backend='retinaface')
            age = int(analysis[0]['age'])
            gender = analysis[0]['gender']
            gender_short = 'M' if gender['Man'] > gender['Woman'] else 'F'
        except Exception as e:
            print(f"⚠️ Failed to analyze face {key}: {e}")
            age = "N/A"

        # Draw box + age label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            img,
            f"{gender_short}: {age}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        ages.append(age)
        genders.append(gender)

    # Show result
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Detected Faces: {len(faces)}")
    plt.show()

    print(f"✅ Number of faces detected: {len(faces)}")
    print(f"✅ Detected ages: {ages}")
    print(f"✅ Detected genders: {genders}")
    
    # Save result
    save_path = generate_output_path(image_path)
    cv2.imwrite(save_path, img)
    print(f"💾 Saved output image to: {save_path}")

def generate_output_path(original_path):
    base, ext = os.path.splitext(original_path)
    return f"{base}_with_ages{ext}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_faces.py path/to/image.jpg")
    else:
        main(sys.argv[1])
