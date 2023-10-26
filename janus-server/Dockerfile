FROM ubuntu:bionic

RUN echo ' \n\
APT::Periodic::Update-Package-Lists "0";\n\
APT::Periodic::Unattended-Upgrade "1";\n'\
> /etc/apt/apt.conf.d/20auto-upgrades

RUN set -x \
    && apt-get update && apt-get install -y aptitude libmicrohttpd-dev libjansson-dev \
	libssl-dev libsofia-sip-ua-dev libglib2.0-dev \
	libopus-dev libogg-dev libcurl4-openssl-dev liblua5.3-dev \
	libconfig-dev pkg-config libtool automake \
    && apt-get install -y build-essential snapd aptitude git wget golang \
    python3 python3-pip python3-setuptools python3-wheel ninja-build \
    libgstreamer1.0-dev libgirepository1.0-dev libunwind-dev apt-utils

RUN set -x \
    && aptitude install -y doxygen graphviz

RUN set -x \
    && wget https://cmake.org/files/v3.16/cmake-3.16.2.tar.gz \
    && tar -xvzf cmake-3.16.2.tar.gz \
    && cd cmake-3.16.2 \
    && ./bootstrap --prefix=/usr/local \
    && make && make install



RUN set -x \
    && apt-get install -y python3-pip \
    && pip3 install scikit-build \
    && pip3 install meson \
    && pip3 install ninja
RUN set -x \
    && git clone https://gitlab.freedesktop.org/libnice/libnice \
    && cd libnice \
    && meson --prefix=/usr build && ninja -C build &&  ninja -C build install
RUN  set -x \
    && cd  \
    && wget https://github.com/cisco/libsrtp/archive/v2.2.0.tar.gz \
    && tar xfv v2.2.0.tar.gz  \
    && cd libsrtp-2.2.0/ \
    && ./configure --prefix=/usr --enable-openssl \
    && make shared_library && make install

#RUN  set -x \
#     && apt-get install -y sudo \
#     && git clone https://boringssl.googlesource.com/boringssl \
#     && sed -i s/" -Werror"//g CMakeLists.txt \
#     && mkdir -p build \
#     && cd build \
#     && cmake -DCMAKE_CXX_FLAGS="-lrt" .. \
#     && make all \
#     && cd .. \
#     && sudo mkdir -p /opt/boringssl \
#     && sudo cp -R include /opt/boringssl/ \
#     && sudo mkdir -p /opt/boringssl/lib \
#     && sudo cp build/ssl/libssl.a /opt/boringssl/lib/ \
#     && sudo cp build/crypto/libcrypto.a /opt/boringssl/lib/

RUN  set -x \
    && git clone https://github.com/sctplab/usrsctp \
    && cd usrsctp \
    && ./bootstrap \
    && ./configure --prefix=/usr --disable-programs --disable-inet --disable-inet6 \
    && make && make install
RUN  set -x \
    && git clone https://github.com/warmcat/libwebsockets.git \
    && cd libwebsockets \
    && mkdir build \
    && cd build \
    && cmake -DLWS_MAX_SMP=1 -DLWS_WITHOUT_EXTENSIONS=0 -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_C_FLAGS="-fpic" .. \
    && make && make install
RUN set -x \
    && git clone https://github.com/eclipse/paho.mqtt.c.git \
    && cd paho.mqtt.c \
    && make && make install
#RUN set -x \
#    && git clone https://github.com/alanxz/rabbitmq-c \
#    && cd rabbitmq-c \
#    && git submodule init \
#    && git submodule update \
#    && mkdir build && cd build \
#    && cmake -DCMAKE_INSTALL_PREFIX=/usr .. \
#    && make && make install

#RUN set -x \
#    && git clone https://gitlab.freedesktop.org/libnice/libnice \
#    && cd libnice \
#    && meson --prefix=/usr build && ninja -C build && ninja -C build install
RUN export JANUS_WITH_POSTPROCESSING
RUN  set -x \
    && git clone https://github.com/meetecho/janus-gateway.git \
    && cd janus-gateway \
    && sh autogen.sh \
    && ./configure --prefix=/opt/janus \
    && make \
    && make install \
    && make configs
RUN  set -x \
    && rm -rf /libwebsockets \
    && rm -rf /janus-gateway \
    && rm -rf /boringssl

RUN set -x \
    apt-get install -y vim sudo

RUN adduser --disabled-password --gecos '' janus
USER janus
RUN #/opt/janus/bin/janus -v
CMD ["/opt/janus/bin/janus"]