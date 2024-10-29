FROM arm64v8/python:3.10

RUN apt-get -y update
RUN apt-get -y upgrade

#Update packages installed in the image
# RUN apt-get update -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# To fix gcc error during build.
# RUN apt install gcc -y
RUN apt-get install build-essential -y

#Change our working directory to app folder
WORKDIR /app-docker
COPY requirements.txt requirements.txt

# Update pip to the latest version
RUN pip3 install --upgrade pip

#Install all the packages needed to run our web app
RUN pip3 install -r requirements.txt

# Add every files and folder into the app folder
COPY . .
# Expose port 5000 for http communication
EXPOSE 5000
# Run gunicorn web server and binds it to the port
CMD python3 -m flask run --host=0.0.0.0