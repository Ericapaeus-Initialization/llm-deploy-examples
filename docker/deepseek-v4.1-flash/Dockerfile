FROM lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5

# Fixed community implementation, including its licenses and source attribution.
ADD https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/archive/45f538a18569420721e00e37353f9a7e1af7e5da.tar.gz /tmp/dsv41-source.tar.gz
RUN mkdir -p /opt/dsv41 && tar -xzf /tmp/dsv41-source.tar.gz --strip-components=1 -C /opt/dsv41 \
    && g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
       /opt/dsv41/adapter/row_store.cpp -o /opt/dsv41/adapter/librow_store.so \
    && cp /opt/dsv41/runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
ENV PYTHONPATH=/opt/dsv41/adapter
WORKDIR /opt/dsv41
COPY launch.py /opt/dsv41/launch.py
ENTRYPOINT ["python3", "-u", "/opt/dsv41/launch.py"]
CMD ["/root/config.yaml"]
