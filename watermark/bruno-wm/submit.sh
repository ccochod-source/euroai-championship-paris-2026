#!/bin/bash
# usage: submit.sh <file.pt>   (run on the login node; key read from ~/.hack_api_key)
curl -s -m 180 -X POST "http://34.147.229.55/submit/34-watermark-localisation-image" -H "X-API-Key: $(cat ~/.hack_api_key)" -F "file=@$1"; echo
