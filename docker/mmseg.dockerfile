FROM youdaoyzbx/ymir-executor:ymir2.1.0-mmseg-cu111-tmi

COPY . /modAL
RUN git clone https://github.com/open-mmlab/mmsegmentation.git -b v0.30.0 /mmseg
RUN pip install -U numpy && pip install omegaconf pandas==1.1.0 scikit-learn
