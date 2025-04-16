# detect_faces
# 1. Install deepface locally
pip install https://github.com/serengil/deepface.git

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run the program
python detect_faces.py images\img1.jpg

# 4. Build Docker
docker build -t detect_faces:latest .

# 5. Run Docker
docker run --rm -it detect_faces:latest 
