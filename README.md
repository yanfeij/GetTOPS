python scrips to download opacity from TOPS website directly. 
It is based on:
https://github.com/tboudreaux/pytopsscrape

TOPsTool.py includes all the functions.

QueryTOPs.ipynb shows how to use the function to download the data.

load_opacity.ipynb can be used to convert the data to tables that can be used in simulations.

If you get errors such as "certificate verification failed", you may need to load the certificate file to your python manually. The commands are given at the top of QueryTOPs.ipynb. You will need to have openssl installed. Remove the command sign # and run the following commands once:

!openssl s_client -showcerts -connect aphysics2.lanl.gov:443 </dev/null 2>/dev/null  | awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/' > tops.crt
cert_file=certifi.where()
source_file="tops.crt"
import subprocess
command = f"cat {source_file} >> {cert_file}"
subprocess.run(command, shell=True, check=True)


 
