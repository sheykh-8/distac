# Use an official PyTorch image as a base
FROM pytorch/torchserve:latest

# Set the working directory in the container
WORKDIR /home/model-server

# Copy the model archive to the model store in the container
COPY model_store/tacotron_encoder.mar /home/model-server/model-store/

# Copy the custom handler
COPY tacotron_handler.py /home/model-server/tacotron_handler.py

# Expose the default TorchServe ports
EXPOSE 8080 8081

# Define environment variables
ENV MODEL_STORE=/home/model-server/model-store
# ENV TS_CONFIG_FILE=/home/model-server/config.properties

# Start TorchServe when the container launches
CMD ["torchserve", "--start", "--model-store", "model-store", "--models", "tacotron_encoder=tacotron_encoder.mar"]
