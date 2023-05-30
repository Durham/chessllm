import requests
payload = {'context':'1-0 1200 1200 1.','beam_size':1, 'action':'generate', 'length':5}
result = requests.post("http://127.0.0.1:8082/", json=payload)
print(results.text)

