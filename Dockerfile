FROM python:3.8-slim
COPY . /app
COPY ./lib/ /app
COPY ./images/ /app/images/
COPY ./data/ /app/data/
COPY ./data/Kenya/ /app/data/Kenya/
WORKDIR /app
# RUN apt-get update && \
#     apt-get install -y build-essential  && \
#     apt-get install -y curl
# RUN apt-get update && \
#     apt-get install -y apt-transport-https && \
#     curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
#     curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
#     apt-get update && \
#     ACCEPT_EULA=Y apt-get install msodbcsql17 unixodbc-dev -y
RUN pip install -r requirements.txt
RUN pip install psm-0.0.10-py3-none-any.whl
EXPOSE 80
RUN mkdir ~/.streamlit
RUN mkdir ~/filestore
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]