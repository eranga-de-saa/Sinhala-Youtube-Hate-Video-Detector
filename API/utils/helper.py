#-*- coding: utf-8 -*-
import urllib

import requests

# proxy = {"http": "http://192.248.8.97:3128"}
# proxies=urllib.request.getproxies()


def openURL(URL, params):
    r = requests.get(URL + "?", params=params)
    return r.text