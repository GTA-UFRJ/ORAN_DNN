import io
import numpy as np
import torch


'''
sample: a couple Inphase Quadrature (float, float)
chunk_size: the sample number per chunk
chunk: the numpy array we deal with
chunk_num: total number of chunks to serve
chunk_offset: the number of initial chunks to skip (for creating sub datasets)
stride: the sample distance between chunk starting sample (default=chunk_size)

dtype '<f4' means 'little endian floating variable of 4 bytes'
'''
class IQData(object):
    def __init__(self, filename, label, chunk_size, chunk_num, chunk_offset=0, stride=0):
        self.filename = filename
        self.data = np.memmap(filename, dtype='<f4', mode='r')
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.label = torch.tensor(label, dtype=torch.long)
        self.chunk_offset = chunk_offset
        self.normalization = None
        self.stride = stride
        if self.stride <= 0:
            self.stride = chunk_size

    def __len__(self):
        return self.chunk_num

    def fetch_record(self, idx):
        start = (idx+self.chunk_offset)*(self.stride*2)
        end = start + self.chunk_size*2
        d = self.data[start:end]
        return d

    def __getitem__(self, idx):
        if idx < len(self):
            d = self.fetch_record(idx)
            d = np.copy(d)
            d = np.transpose(d.reshape((self.chunk_size, 2)))
            d = torch.from_numpy(d)
            if self.normalization:
                d = (d - self.normalization[0]) / self.normalization[1]
            return (d, self.label)
        else:
            raise IndexError

    def normalize(self, mean, std):
        self.normalization = (mean.unsqueeze(-1), std.unsqueeze(-1))


class IQDataset(object):
    def __init__(self, data_folder=".", chunk_size=20000, stride=0, subset='train'):
        self.label = {0: 'clear', 1: 'LTE', 2: 'WiFi', 3: 'other'}
        self.dataset = []
        if stride <= 0:
            stride = chunk_size

        if subset == 'train':
            chunks_per_dataset = (1200000000-chunk_size)//stride +1
            #chunks_per_dataset = 100
            self.dataset.append(IQData(data_folder + "/CLEAR.bin", label=0, stride=6666,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset*3))
            self.dataset.append(IQData(data_folder + "/LTE_FLOOD.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData(data_folder + "/LTE_1M.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData(data_folder + "/LTE_ZT.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData(data_folder + "/WIFI_FLOOD.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData(data_folder + "/WIFI_1M.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))
            self.dataset.append(IQData(data_folder + "/WIFI_ZT.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset))



        elif subset == 'validation':
            offset = (1200000000-chunk_size)//stride +1
            chunks_per_dataset = (600000000-chunk_size)//stride +1
            #chunks_per_dataset = 100
            self.dataset.append(IQData(data_folder + "/CLEAR.bin", label=0, stride=6666,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset*3,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/LTE_FLOOD.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/LTE_1M.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/LTE_ZT.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/WIFI_FLOOD.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/WIFI_1M.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/WIFI_ZT.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))

        elif subset == 'test':
            offset = (1800000000-chunk_size)//stride +1
            chunks_per_dataset = (600000000-chunk_size)//stride +1
            #chunks_per_dataset = 100
            self.dataset.append(IQData(data_folder + "/CLEAR.bin", label=0, stride=6666,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset*3,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/LTE_FLOOD.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/LTE_1M.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/LTE_ZT.bin", label=1, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/WIFI_FLOOD.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/WIFI_1M.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            self.dataset.append(IQData(data_folder + "/WIFI_ZT.bin", label=2, stride=stride,
                                       chunk_size=chunk_size, chunk_num=chunks_per_dataset,
                                       chunk_offset=offset))
            #self.dataset.append(IQData(data_folder + "/ofdm_benchmark.bin", label=3, stride=6666,
            #                           chunk_size=chunk_size, chunk_num=chunks_per_dataset*3,
            #                           chunk_offset=0))

        self.chunks_per_dataset = chunks_per_dataset
        self.chunk_size = chunk_size

    def __len__(self):
        l = 0
        for d in self.dataset:
            l += len(d)
        return l

    def __getitem__(self, idx):
        if idx < len(self):
            ds = 0
            cnt = idx
            while cnt >= len(self.dataset[ds]):
                cnt -= len(self.dataset[ds])
                ds += 1
                if ds >= len(self.dataset):
                    raise IndexError
            idx = cnt
            d = self.dataset[ds][idx]
            return d
        else:
            raise IndexError

    def stats(self):
        '''
        returns a tuple of tensors indicating the mean and standard deviation
        for each channel (I and Q)
        '''
        mean = torch.tensor([0, 0])
        var = torch.tensor([0, 0])
        n = 0
        for chunk, label in self:
            m = chunk.mean(dim=1)
            v = chunk.var(dim=1)

            nm = mean*n/(n+self.chunk_size)
            nm += m*self.chunk_size/(n+self.chunk_size)

            nv = var*(n-1)/(n+self.chunk_size-1)
            nv += torch.pow(mean, 2)*n/(n+self.chunk_size-1)
            nv += v*(self.chunk_size-1)/(n+self.chunk_size-1)
            nv += torch.pow(m, 2)*self.chunk_size/(n+self.chunk_size-1)
            nv -= torch.pow(nm, 2)*(n+self.chunk_size)/(n+self.chunk_size-1)

            mean = nm
            var = nv
            n += self.chunk_size

        return (mean, var.sqrt())

    def normalize(self, mean, std):
        '''
        mean and std are expected to be torch tensors (as returned from
        stats())
        '''
        for d in self.dataset:
            d.normalize(mean, std)


if __name__ == "__main__":
    '''
    I/Q sample rate is 20M. For a chunk of 1ms we take chunk_size of 20K.
    In a minute we collect 20M*60 = 1.2G samples (total of 1.2G*2*4=9.6GB)
    num of chunks in a minute = 1.2G/20k = 60k (if chunks do not overlap, i.e.,
    stride=chunk_size).
    '''
    d = IQData("clear.bin", label='clear', chunk_size=20000, chunk_num=60000)
    print(d[0])