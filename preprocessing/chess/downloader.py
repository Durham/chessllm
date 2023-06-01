import os
import requests
from bs4 import BeautifulSoup
import re
import zstandard
from parser import parse_chess_file

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            i = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                print(i, " ", end='\r')
    return local_filename

def decompress_file(input_file):
    output_file = input_file.replace('.zst', '')
    dctx = zstandard.ZstdDecompressor()
    with open(input_file, 'rb') as ifh, open(output_file, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)



if __name__=="__main__":
 url = "https://database.lichess.org/#standard_games"
 base_url = "https://database.lichess.org/"

 response = requests.get(url)
 soup = BeautifulSoup(response.text, "html.parser")

 for link in soup.find_all('a', href=re.compile("^standard/.*\.pgn.zst$")):
    file_url = base_url + link['href']
    print(f'Downloading {file_url}')
    filename = download_file(file_url)
    print(f'Decompressing {filename}')
    local_filename = decompress_file(filename)
    os.remove(filename)  # remove the compressed file after decompression
    parse_chess_file(local_filename)
    os.remove(local_filename)  # remove the compressed file after decompression
    print(f'Done with {filename}')

