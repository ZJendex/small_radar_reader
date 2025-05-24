import listener
# import organizer
import sys
import pickle
import glob
import socket
from pathlib import Path


filename = sys.argv[1]
fileroot = filename[:-4]

filename_without_dir = Path(filename).name
#remove the foot 
filename_without_dir = filename_without_dir.split('.')[0]

obj = listener.UDPListener(fileroot=fileroot)
host = "127.0.0.2"
port = 12346
capture_directory = filename_without_dir  
message = f"1 {capture_directory}"
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

input("Press Enter to continue...")

try:
    client.connect((host, port))
    client.sendall(message.encode('utf-8')) 
finally:
    client.close()
    
all_data = obj.read()

print("Start time: ", all_data[3])
print("End time: ", all_data[4])

with open(filename, 'wb') as f:
	pickle.dump(all_data, f)

print("Storing collected files in ", filename)


# print("Start time: ", all_data[0])
# print("End time: ", all_data[1])

# with open(fileroot + '_timestamp.pkl', 'wb') as f:
# 	pickle.dump(all_data, f)

# print("Storing collected files in ", fileroot)	