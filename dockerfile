# use lighweight python 

FROM python 3.12-slim

# set working directory
WORKDIR /app

# copy files
COPY . /app 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
