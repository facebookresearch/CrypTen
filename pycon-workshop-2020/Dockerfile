# This pulls pytorch image with Python 3.7
FROM pytorch/pytorch

RUN  pip install crypten
RUN  pip install jupyter

COPY Intro.ipynb Intro.ipynb

# start jupyter notebook
# Tini operates as a process subreaper for jupyter to prevent kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
