# distac
A simple distributed version of the famous TTS model tacotron for scalablility on the cloud

This project divides a pretrained Tacotron model to 4 different parts: Encoder, Decoder, Postnet and Vocoder. With the help of an orchestrator we can use this model to inference speech from text and apply load balancing to build a scalable model with less resources than we would need if we were to scale the entire model.

## Building images and usage

You can build the images by executing this command for each part of the model from the root of the project.

```bash
docker build -f Dockerfiles/[model-part].Dockerfile -t [you_username]/[model-part]:latest .
```

It's also possible to start this repository using docker compose by running this command in the root of the repository: 

```bash
docker compose up
``` 


## inference:

To generate speech from the input text, considering all the services are up and running (with the orchestrator running on port 8080) you can generate speech by sending the following request:

```bash
curl -X POST http://localhost:8080/process -H "Content-Type: application/json" -d '{"text": "Only three stars are born in the milkyway each year"}' -o result.wav
```