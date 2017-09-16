###########
#
#      DOCKER CONTAINER FOR KLUSTR PROJECT:
# MAINTAINED BY: Lamtharn (Hanoi) Hantrakul & Avneesh Sarwate - GTCMT
#
# Please add new pip/apt installs in this block. Don't forget a "&& \" at the end
# of all non-final lines. Thanks!
#
###########

### Pull official Anaconda Python Distribution 4.4.0 (which has python3.6)
FROM continuumio/anaconda:4.4.0

### Install Custom Libraries

# Install Librosa for audio analysis
RUN conda install -c conda-forge librosa  && \

# Install Jupyter
conda install Jupyter && \

# Install scikit-learn
conda install -c anaconda scikit-learn && \

# Install tqdm
pip install tqdm 
