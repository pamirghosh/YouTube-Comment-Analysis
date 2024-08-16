# Use the official Python base image
FROM python:3.12.2

# Install dependencies required by Ollama and other tools
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -sSf https://ollama.com/install.sh | sh

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the necessary ports
EXPOSE 8501 11400

# Run Ollama in the background and then start your application
CMD ollama serve & streamlit run app.py
