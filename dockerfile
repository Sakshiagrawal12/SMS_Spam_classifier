<<<<<<< HEAD

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
=======
# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all files from project folder to container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords (required by your program)
RUN python -m nltk.downloader stopwords

# Run your program
CMD ["python", "sms_Spam_classifier.py"]
>>>>>>> e40ad7055147c83ebfcbbbb78ae40fe0f4afc39f
