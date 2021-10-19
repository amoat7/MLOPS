# Use an exiting image as a base
FROM alpine


# Download and install a dependency 
RUN apk add --update redis

RUN apk add --update gcc

# Tell the image what to do when is starts as a container
CMD ["redis-server"]

# docker build -t <docker_id/repo(project_name):version> . - build docker file
