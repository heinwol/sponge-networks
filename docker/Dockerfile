FROM jupyter/minimal-notebook:latest

USER root

ENV GRANT_SUDO=yes
ENV JUPYTER_TOKEN='0000'

RUN apt-get update -y
# RUN apt-get upgrade -y
RUN apt-get install -y fish neovim graphviz libgraphviz-dev git gcc make cmake 

COPY . /home/jovyan/sponge-networks

RUN pip install poetry \
    && pip install -U jsonschema \
    && chown -R jovyan /home/jovyan

USER jovyan

# actual project
RUN cd /home/jovyan/sponge-networks \
    && poetry build \
    && pip install --user dist/*.whl matplotlib array_to_latex cairosvg

# COPY jupyter-config .

# https://discourse.jupyter.org/t/customizing-shell-launched-by-terminal/3412
# RUN cat jupyter_notebook_config.py.txt >> /etc/jupyter/jupyter_notebook_config.py

WORKDIR "${HOME}/Documents"

# CMD [ "fish", "-i" ]

# docker build . -t my-jupyter-image
# docker run --name my-datasciense-notebook -it --user root -p 8889:8888 -v ~/Documents:/home/jovyan/Documents my-jupyter-image
