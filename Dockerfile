FROM python:3.10-slim-buster
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip install tf-keras
COPY . .
CMD ["python", "detect_faces.py", "images/img3.jpg"]