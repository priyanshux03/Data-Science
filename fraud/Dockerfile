FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
