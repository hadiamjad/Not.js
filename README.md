# TrackerSift
`TrackerSift`, a powerful tool designed for security researchers to easily crawl websites and capture various API calls, network requests, and dynamic features of JavaScript, including stack traces. `TrakerSift` uses selenium to automatically crawl and navigate through a website, capturing all relevant information using purposely-built chrome extension. Chrome extension can identify and collect data from various sources, including APIs, database queries, server responses, and user interactions.
With `TrakerSift`, you can easily analyze the behavior of a website's JavaScript code, including the different function calls, event listeners, and error messages. The tool is designed to capture stack traces, which can provide valuable insights into the code execution path and help identify potential issues and vulnerabilities.`TrakerSift` exports your findings in JSON format.

`TrackerSift, Abdul Haddi Amjad, Zubair Shafiq, Muhammad Ali Gulzar, IMC'21`

The published version of the manuscript is avaibable at : [TrackerSift](https://dl.acm.org/doi/abs/10.1145/3487552.3487855)

## Installation
> We are working on docker and soon we will release the DockerFile

Meanwhile follow these:
### Requirements
1. [NodeJS]()
2. Python [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
3. [Xvfb]()

### Steps
#### Crawling
1. Clone this repository and move inside the repo directory using `cd` command
2. Open `two` terminals inside repository directory
---- First Terminal ----
3. In first terminal, run `cd server` and then `node server.js` -- this will start the local-host server at `Port:3000` which communicates with chrome-extension to save the captured data inside `server/output` directory. 
---- Second Terminal ----
4. In second terminal, create `conda` environment using `requirement.txt` -- `conda create --name envname --file requirements.txt`
5. Now if you want to crawl a specific website e.g, `livescore.com` you can add any list of websites mentioned in `ten.csv` which is specified in `sele.py` - `line 20`
6. After specifying the websites, simply activate the conda env and run `python sele.py` -- this will crawl the website and store the respective data in `output/server/{website}/` directory e.g, for `livescore.com` you can find in `output/server/livescore.com/`.
> See `SCHEMA.md` for each file.
#### Labelling network request using EasyList and EasyPrivacy
7. In the same conda env you can run `python label.py` to label the network requests for all crawled websites. This will create `label_request.json` inside each website directory.
#### Generating graph
7. In the same conda env you can run `python graph-plot/main.py` to create graph for each website. This will create `label_request.json` inside each website directory.
#### Extracting features from graph
8. In the same conda env you can run `python graph-plot/makeFeatures.py` to create graph for each website. This will create multiple files inside website directory, see `SCHEMA.md` for each file.

## Research papers that used this instrumentation
1. [TrackerSift](https://arxiv.org/pdf/2302.01182.pdf)
2. [Blocking JavaScript without Breaking the Web](https://arxiv.org/pdf/2302.01182.pdf)

## Citation
@inproceedings{Amjad22TrackerSift,
  title     = {TrackerSift},
  author    = {Abdul Haddi Amjad, Zubair Shafiq, Muhammad Ali Gulzar},
  booktitle = {ACM Internet Measurement Conference (IMC)},
  year      = {2021}
}

## Contact
Please contact [Hadi Amjad](https://hadiamjad.github.io/) if you run into any problems running the code or if you have any questions.