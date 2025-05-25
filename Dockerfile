FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel
WORKDIR /user
COPY . .
RUN pip install --no-cache-dir -r requirements.txt 
COPY . .
CMD ["python","model.py"]