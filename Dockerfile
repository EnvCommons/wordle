FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the NLTK datasets TextArena's Wordle needs into the image so they never
# have to be downloaded at runtime. Combined with the nltk.download no-op in env.py,
# this keeps every session start off the network.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import nltk; [nltk.download(p, download_dir='/usr/share/nltk_data') for p in ('words', 'averaged_perceptron_tagger_eng')]"

COPY . /app/

EXPOSE 8000

CMD ["python", "server.py"]
