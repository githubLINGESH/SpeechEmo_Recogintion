import requests

url = "https://api.play.ht/api/v2/cloned-voices"

api_key='a2d236fb9d6f4f99ab6845003bb2d0fa'

headers = {
    "accept": "application/json",
    "X-User-Id": "nTx2MoNL1tXcxYbuGbuzwg0vNNw1",
    "Authorization": f"Bearer {api_key}",
}

response = requests.get(url , headers= headers)

print(response.text)