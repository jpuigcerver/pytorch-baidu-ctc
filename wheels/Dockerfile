FROM centos:7
LABEL maintainer "Joan Puigcerver <joapuipe@gmail.com>"
ENV DOCKER=1

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl
RUN yum install -y yum-utils centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum install -y devtoolset-3-gcc devtoolset-3-gcc-c++ devtoolset-3-gcc-gfortran devtoolset-3-binutils
ENV PATH=/opt/rh/devtoolset-3/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-3/root/usr/lib64:/opt/rh/devtoolset-3/root/usr/lib:$LD_LIBRARY_PATH

# EPEL for cmake
RUN \
wget http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
rpm -ivh epel-release-latest-7.noarch.rpm && \
rm -f epel-release-latest-7.noarch.rpm;

RUN mkdir -p /tmp/build_image;
COPY *.sh *.py /tmp/build_image/
WORKDIR /tmp/build_image/
RUN ./build_manylinux.sh;

WORKDIR /
RUN rm -rf /tmp/build_image;