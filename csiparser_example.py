import numpy as np
import matplotlib.pyplot as plt


ENDIANESS = 'big'

def signbit_convert(data, maxbit):
    if data & (1 << (maxbit - 1)):
        data -= (1 << maxbit);
    return data;

def ath_processCsiBuffer_online(data, params, res):
    
    blocklen = int.from_bytes(data[:2], byteorder=ENDIANESS)
    nsubc     = int(data[18])
    nr        = int(data[19])
    nc        = int(data[20])    
    if nr != params['Nrx'] or nc != params['Ntx'] or nsubc != params['Nsubcarriers']:
        return False
    if  nr * nc * nsubc * 20 / 8 > blocklen - 27:
        return False
                  
    ath_decodeCSIMatrix(nr, nc, nsubc, res, data)
    return True

def ath_decodeCSIMatrix(nr, nc, nsubc, matrix, data):
    bitmask = (1 << 10) - 1
    idx = 27
    current_data = 0
    bits_left = 0 # process 16 bits at a time

    for k in range(nsubc):
        for nc_idx in range(nc):
            for nr_idx in range(nr):

                if bits_left < 10:
                    h_data = data[idx]
                    idx += 1
                    h_data += (data[idx] << 8);
                    idx += 1
                    current_data += h_data << bits_left
                    bits_left += 16

                # img
                img = float(signbit_convert(current_data & bitmask, 10))
                bits_left -= 10
                current_data = current_data >> 10

                if bits_left < 10:
                    h_data = data[idx]
                    idx += 1
                    h_data += (data[idx] << 8);
                    idx += 1
                    current_data += h_data << bits_left
                    bits_left += 16

                # real
                real = float(signbit_convert(current_data & bitmask, 10)) 
                bits_left -= 10
                current_data = current_data >> 10

                matrix[nr_idx, nc_idx, k] = real + 1j * img

    if nsubc == 114: # for 40mhz need to apply 90deg rotation
        matrix[:,:,57:] = matrix[:,:,57:] * np.exp(-1j * np.pi/2)
        

def ath_parseFile(fpath, params, filepercent=100, limit=100000):
    csilist = np.empty((limit, params['Nrx'], params['Ntx'], params['Nsubcarriers']), dtype=complex)
    count = 0
    with open(fpath, 'rb') as fp:
        data = fp.read()
        l = len(data)
        p = 0
        s = (l/100) * (100 - filepercent)
        while p < l - s and count < limit:
            bl = int.from_bytes(data[p:p+2], byteorder=ENDIANESS)
            ath_processCsiBuffer_online(data[p:], params, csilist[count])
            p += bl + 2
            count+=1
    return csilist[:count]
        
params = {
	'Nrx':3,
	'Ntx':3,
	'Nsubcarriers':114,
}

res = ath_parseFile('CSI', params, filepercent=10)
print(res.shape)


res = np.reshape(res, (res.shape[0], res.shape[1]*res.shape[2]*res.shape[3]))
plt.figure()
plt.plot(np.abs(res.T))
plt.figure()
plt.plot(np.angle(res.T))
plt.show()
