FROM apache/airflow:2.7.1-python3.10

USER root
RUN apt-get update -y && \
    apt-get install build-essential openjdk-11-jdk wget git -y

USER airflow

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY volumes volumes

# RUN wget -P /home/airflow/.local/lib/python3.8/site-packages/pyspark/jars https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar && \
#     wget -P /home/airflow/.local/lib/python3.8/site-packages/pyspark/jars https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
#     wget -P /home/airflow/.local/lib/python3.8/site-packages/pyspark/jars https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.13/3.4.0/spark-hadoop-cloud_2.13-3.4.0.jar

