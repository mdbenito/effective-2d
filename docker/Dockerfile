##############################################################################
# MBD: lvk-dev
#   - See README.md for info
# FIXME: freeze versions for all packages!

FROM quay.io/fenicsproject/stable:2017.1.0
MAINTAINER Miguel de Benito Delgado <m.debenito.d@gmail.com>

ENV home=/home/fenics
ENV src=/home/fenics/lvk
ENV bin=/usr/local/bin

##############################################################################
# Install necessary tools and python packages
#   - Pandoc for PDF export of notebooks (disabled, requires LaTeX)
#   - vim for quick editing inside the container
#   - nbimporter: import IPython notebooks as modules
#   - autograd: Automatic differentiation with numpy support

RUN apt-get update && \
    apt-get install --yes --no-install-recommends vim && \
    rm -rf /var/lib/apt/lists/*
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

##############################################################################
# Setup terminal, editor related stuff and assorted tools
#   - git-prompt.sh for informative terminal prompts in git repos
#   - .inputrc: better navigation of command line history
#   - .vimrc: minimal vim config
#   - .bashrc: extra stuff
#   - fenics-notebook to start a notebook using `docker exec ...`

# Use copy from context to avoid creating a new image for each build:
COPY docker/git-prompt.sh $home/.git-prompt.sh
COPY docker/dot_bashrc_extra /tmp/dot_bashrc_extra
COPY docker/dot_inputrc $home/.inputrc
COPY docker/dot_vimrc $home/.vimrc
COPY docker/fenics-notebook $bin/fenics-notebook
COPY docker/report-server $bin/report-server
COPY docker/WELCOME $home/WELCOME
RUN mkdir -p $src/meshes
RUN mkdir $home/.jupyter
COPY docker/jupyter_notebook_config.json $home/.jupyter/

##############################################################################
# Final setup, fix permissions

RUN /bin/cat /tmp/dot_bashrc_extra >> $home/.bashrc && \
    rm -f /tmp/dot_bashrc_extra && \
    chmod 744 $bin/fenics-notebook $bin/report-server && \
    chown -R fenics.fenics $home/.* $bin/fenics-notebook \
                           $bin/report-server

WORKDIR $home
