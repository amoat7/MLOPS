FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements.txt .

RUN pip install -r requirements.txt && \
	rm requirements.txt


COPY ./app /app

EXPOSE 80

WORKDIR  /app


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

#sudo docker build -t mlepc-fastapi:no-batch .
#sudo docker run --rm -p 80:80 mlepc-fastapi:no-batch
