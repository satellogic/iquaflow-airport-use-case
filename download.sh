#!/bin/bash

TO_PATH=yolov5/datasets && \
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)" && \
wget https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-airport-use-case/AIRPORT2CROPS998cleanth886505.zip -O zipfile.zip && \
chmod 775 ./zipfile.zip && unzip -o ./zipfile.zip && rm ./zipfile.zip