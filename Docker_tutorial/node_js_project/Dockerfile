# Specify the base image
FROM node:alpine

WORKDIR /usr/app

COPY ./package.json ./

# Install some dependencies
RUN npm install 

COPY ./ ./


# Default command
CMD ["npm", "start"]

# docker build -t <dave/simpleweb>
# docker run -p 8080:8080 <image_name> 