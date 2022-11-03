#server.py
import socket

port = 12345  # tcp port number
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', port))

# conf num client it can listen concurrently
s.listen(1)
conn, addr = s.accept()

while True:
    data = conn.recv(1024).decode()
    if not data:
        break # if data is not received break
    print("from connected user: " + str(data))
    data = input('-> ')
    conn.send(data.encode())

conn.close()
