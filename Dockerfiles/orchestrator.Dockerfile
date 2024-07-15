FROM python:3.8-slim

WORKDIR /app

COPY model/orchestrator.py /app

RUN pip install Flask requests gunicorn

EXPOSE 80

# CMD ["gunicorn", "orchestrator_app.py"]
CMD ["gunicorn", "-w", "2", "--timeout", "1000", "-b", "0.0.0.0:80", "orchestrator:app", "--log-level", "debug"]