FROM ubuntu:14.04

MAINTAINER ataber
RUN apt-get update

RUN apt-get install python3
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade cython
RUN pip3 install --upgrade numpy
RUN pip3 install --upgrade flask
RUN pip3 install --upgrade flask-CORS
RUN pip3 install --upgrade PyMCubes
#RUN pip3 install --upgrade scikit-image
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py3-none-any.whl
RUN pip3 install --upgrade sphinx
RUN pip3 install --upgrade sphinx_rtd_theme
RUN apt-get install -y curl
