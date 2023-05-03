import urllib.request
import json
import base64
import hmac
import time

APIURL = "https://open-api.bingx.com"
APIKEY = "IKEUMhjgJEDuZOjkN5oJOZvFUIivqaNJp9FaMslx6SzcyasbJ3O8o0UvgOki762XSMKGXRMPi1zGSF6QgDg"
SECRETKEY = "UkfMeuSJsZvQrrvfEkJbLz7207eWAG4wQeMcjljtJqRUolex4Thn7tH7JjsQiuFRBuMbewUVCjk8WfcuA"

def genSignature(paramsStr:str):
    return hmac.new(SECRETKEY.encode("utf-8"),
        paramsStr.encode("utf-8"), digestmod="sha256").digest()

def post(url, body):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0',
        'X-BX-APIKEY': APIKEY,
    }, method="GET")
    return urllib.request.urlopen(req).read()


def getURL(contract_url, param_dict):
    paramsStr = "&".join(["%s=%s" % (k, param_dict[k]) for k in param_dict])
    paramsStr += "&signature=" + genSignature(paramsStr).hex()
    url = "%s?%s" % (contract_url, paramsStr)
    return url, paramsStr

def getBalance():
    paramsMap = {
        "timestamp": int(time.time()*1000),
        "symbol":"BTC-USDT",
        "side":"LONG",
        "leverage":6,
    }
    paramsStr = "&".join(["%s=%s" % (k, paramsMap[k]) for k in paramsMap])
    paramsStr += "&signature=" + genSignature(paramsStr).hex()
    url = "%s/openApi/swap/v2/user/balance?%s" % (APIURL, paramsStr)
    return post(url, paramsStr)

def getContracts():
    paramsMap = {
        # "timestamp": int(time.time()*1000),
        # "symbol":"BTC-USDT",
        # "side":"LONG",
        # "leverage":6,
    }
    contract_url = "%s/openApi/swap/v2/quote/contracts"%APIURL
    url, paramsStr = getURL(contract_url, paramsMap)
    return post(url, paramsStr)

def getPrice(pair:str, timestep=None):
    if timestep==None or type(timestep)!=int:
        timestep= int(time.time()*1000)
    paramsMap = {
       "symbol":pair.upper(),
       "timestamp":timestep,
        # "symbol":"BTC-USDT",
        # "side":"LONG",
        # "leverage":6,
    }
    contract_url = "%s/openApi/swap/v2/quote/price"%APIURL
    url, paramsStr = getURL(contract_url, paramsMap)
    return post(url, paramsStr)

def main():
    print(getPrice("BTC-USDT", 1682567426097))

if __name__ == "__main__":
        main()