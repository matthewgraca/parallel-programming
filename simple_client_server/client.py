# client.py
import socket

port = 12345 # Same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', port))
message = input('-> ')

while message.lower().strip() != 'bye':
    s.send(message.encode())
    data = s.recv(1024).decode()
    print('Received from server: ' + data)
    message = input('-> ')

s.close()
