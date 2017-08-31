FROM centos:7

RUN yum install epel-release -y
RUN yum update -y

RUN yum install -y python-pip && yum clean all
RUN pip install --upgrade pip

RUN pip install numpy
RUN pip install tensorflow
RUN pip install pyserial

VOLUME ["/home/deeplearning"]
WORKDIR /home/deeplearning

EXPOSE 8000
