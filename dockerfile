# use lightweight python 
FROM python:3.12-slim

# set working directory  
WORKDIR /app

# copy requirements first (for caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the code
COPY . .

# expose port
EXPOSE 8000

# run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]