import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1000)
    print('TCP_MAXSEG supported')
except Exception as e:
    print('Error:', e)
