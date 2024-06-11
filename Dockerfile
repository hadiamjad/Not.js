FROM joyzoursky/python-chromedriver
USER root
RUN ls

RUN apt-get update
RUN apt install -y nodejs
RUN apt-get install -y tmux
RUN apt-get install -y graphviz
RUN apt-get install -y graphviz-dev
RUN apt install -y xvfb

RUN pip3 install numpy
RUN pip3 install pandas
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
RUN pip install joblib
RUN pip install graphviz
RUN pip install networkx
RUN pip install pygraphviz

WORKDIR /Crawler
COPY . /Crawler
CMD ["/bin/bash"]