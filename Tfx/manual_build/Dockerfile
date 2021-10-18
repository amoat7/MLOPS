# install base image
# Create working dir 
# Copy files 
# install libraries 

# docker build -t automated-image:1.0 .
# sudo docker run -it --rm -p 5000:5000 automated-image:1.0
# sudo docker run -it --network=host --rm -p 5000:5000 automated-image:1.0


FROM python 
WORKDIR /app
COPY hello.py . 
COPY start_app.sh . 
RUN pip install Flask
CMD ["bash", "start_app.sh"]

