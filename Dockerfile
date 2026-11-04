FROM python:3.12-slim
WORKDIR /app
COPY requirements-service.txt .
# Optional corporate-proxy CA: docker build --secret id=ca,src=ca.crt --build-arg PIP_CERT=/ca.crt .
ARG PIP_CERT
RUN --mount=type=secret,id=ca,target=/ca.crt,required=false \
    pip install --no-cache-dir ${PIP_CERT:+--cert $PIP_CERT} -r requirements-service.txt
COPY service/ service/
USER nobody
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
