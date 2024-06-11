#### Crawling
1. `logs.txt` -- total number of websites that crawled successfully excluding unstable and not reachable websites
2. `server/output/cookie_storage.json` has object on each line as follows
    ```
    {
        "top_level_url": "Website" 
        "function": "storage_getter/storage_setter/cookie_getter/cookie_setter",
        "storage":"{key:value} pairs"
        "stack": "while the top of the stack trace includes intercepted extension function underneath shows the sequence of website JS involved."
    }
    ```
3. `server/output/eventget.json` has object on each line as follows
    ```
    {
        "top_level_url": "Website"
        "event": "getAttribute",
        "name":"name of element"
        "this": "the complete element object"
        "stack": "while the top of the stack trace includes intercepted extension function underneath shows the sequence of website JS involved."
    }
    ```
4. `server/output/eventset.json` has object on each line as follows
    ```
    {
        "top_level_url": "Website"
        "event": "setAttribute/addEventListener/removeEventListener",
        ...(different parameter for each even type)
        "stack": "while the top of the stack trace includes intercepted extension function underneath shows the sequence of website JS involved."
    }
    ```
5. `server/output/request.json` has object on each line as follows with following schema 
[Network.requestWillBeSent](https://chromedevtools.github.io/devtools-protocol/tot/Network/#event-requestWillBeSent)

6. `server/output/responses.json` has object on each line as follows with following schema 
[Network.responseRecieved](https://chromedevtools.github.io/devtools-protocol/tot/Network/#event-responseReceived)

7. `server/output/responses.json` has object on each line as follows with following schema 
[Network.requestWillBeSentExtraInfo](https://chromedevtools.github.io/devtools-protocol/tot/Network/#event-requestWillBeSentExtraInfo)

#### Labelling network request using EasyList and EasyPrivacy
8. `server/output/label_request.json` has object on each line as follows with following schema 
[Network.requestWillBeSent](https://chromedevtools.github.io/devtools-protocol/tot/Network/#event-requestWillBeSent) along with three extra flags:
- `easylistflag`
- `easyprivacylistflag`
- `ancestorflag`
these three a 0/1 flags and if either of these are set then request is tracking.
#### Generating graph
> Will be updated soon!
#### Extracting features from graph
> Will be updated soon!
