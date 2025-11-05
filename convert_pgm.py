# mostrar_pgm.py
import matplotlib.pyplot as plt
import numpy as np

with open('hough_output.pgm', 'rb') as f:
    assert f.readline().strip() == b'P5'
    dims = f.readline()
    while dims.startswith(b'#'):  # saltar comentarios
        dims = f.readline()
    w, h = map(int, dims.split())
    maxval = int(f.readline())
    data = np.frombuffer(f.read(w*h), dtype=np.uint8).reshape((h, w))

plt.imshow(data, cmap='gray', aspect='auto')
plt.title('Hough accumulator (theta vertical, r horizontal)')
plt.xlabel('r index (0..%d)' % (w-1))
plt.ylabel('theta bin (0..%d)' % (h-1))
plt.show()