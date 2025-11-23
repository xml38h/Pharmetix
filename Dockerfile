# RDKit official docker image
FROM rdkit/rdkit:latest

# working directory
WORKDIR /app

# install required python packages
COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files (app.py, logic.py, templates/, static/, etc.)
COPY . .

# HuggingFace uses PORT env variable
ENV PORT=7860
EXPOSE 7860

# start Flask app using gunicorn
CMD ["bash", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} app:app"]
