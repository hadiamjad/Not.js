FROM joyzoursky/python-chromedriver
USER root
RUN ls

RUN apt-get update
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y tree
RUN apt-get install -y tmux
RUN apt-get install -y graphviz
RUN apt-get install -y graphviz-dev
RUN apt install -y xvfb

RUN pip3 install numpy==1.25.2
RUN pip3 install pandas
RUN pip install scikit-learn==1.3.0
RUN pip3 install adblockparser
RUN pip3 install openpyxl
RUN pip3 install pyvirtualdisplay
RUN pip3 install selenium
RUN pip3 install seaborn
RUN pip3 install tldextract
RUN pip3 install webdriver-manager
RUN pip3 install matplotlib
RUN pip3 install xlrd
RUN pip3 install beautifulsoup4
RUN pip3 install httpx
RUN pip3 install joblib
RUN pip3 install graphviz
RUN pip3 install networkx
RUN pip3 install pygraphviz
RUN pip3 install gdown

WORKDIR /Crawler
COPY . /Crawler
CMD ["/bin/bash"]