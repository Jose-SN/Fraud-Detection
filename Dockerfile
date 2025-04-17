FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
# Expose the port the app runs on
EXPOSE 8000 
# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
