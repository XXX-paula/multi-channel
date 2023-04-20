import torch
import math
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from torch.autograd import Variable
from torch.nn import functional as tf
from utils import Linear
from torch.nn import init


#网络结构
class GCLSTM(nn.Module):
    def __init__(self):
        super(GCLSTM, self).__init__()
        self.hidden_size = 300
        self.encoder_cell1 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell2 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell3 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell4 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell5 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell6 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell7 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell8 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell9 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell10 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell11 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_cell12 = nn.LSTMCell(8, self.hidden_size, bias=True)
        self.encoder_target = nn.LSTMCell(5, self.hidden_size, bias=True)
        self.sigma = nn.Parameter(torch.FloatTensor(1))
        self.sigmas = nn.Parameter(torch.FloatTensor(1))
        self.wp = nn.Parameter(torch.FloatTensor(300, 600))
        self.bp = nn.Parameter(torch.FloatTensor(300, 1))
        self.ff = nn.Parameter(torch.FloatTensor(100, 13))
        self.bff = nn.Parameter(torch.FloatTensor(100, 1))
        self.Attention1 = nn.Parameter(torch.FloatTensor(600, 200))
        self.biasa1 = nn.Parameter(torch.FloatTensor(200))
        self.Attention2 = nn.Parameter(torch.FloatTensor(202, 1))
        self.biasa2 = nn.Parameter(torch.FloatTensor(1))
        self.a = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.fuse1 = nn.Parameter(torch.FloatTensor(400, 200))
        self.biasf = nn.Parameter(torch.FloatTensor(200))
        self.fuse2 = nn.Parameter(torch.FloatTensor(600, 200))
        self.biasf2 = nn.Parameter(torch.FloatTensor(200))
        # self.Wout = nn.Parameter(torch.FloatTensor(200, 1))
        # self.biasout = nn.Parameter(torch.FloatTensor(1))
        self.Wout = nn.Parameter(torch.FloatTensor(200, 1))
        self.biasout = nn.Parameter(torch.FloatTensor(1))

        self.reset_parameters()
    def reset_parameters(self):
        for weight in self.parameters():
            init.uniform_(weight, -0.1, 0.1)

    #输入数据格式转化
    def input_transform(self, x):
        local_inputs, labels, extras = x
        local_inputs = local_inputs.permute(1, 0, 2, 3) #(12h, batch, 28features, 11站点)
        labels = labels.permute(1, 0, 2, 3) #(12h, batch, 1features, 1)
        extras = extras.permute(1, 0, 2, 3) #(12h, batch, 21features, 1站点)
        n_input_encoder = local_inputs.data.size(2) #28features
        batch_size = local_inputs.data.size(1)
        extra_size = extras.data.size(2)
        _local_inputs = local_inputs.contiguous().view(-1, n_input_encoder, 12) #(12*batch,28,11)
        _local_inputs = torch.split(_local_inputs, batch_size, 0) #12个数组 每个数组（batch，28，11）
        encoder_inputs = _local_inputs
        _labels = labels.contiguous().view(-1, 1)
        _labels = torch.split(_labels, batch_size, 0)
        _extras = extras.contiguous().view(-1, extra_size, 1)  # (12*batch,21,1)
        _extras = torch.split(_extras, batch_size, 0) #(12个数组，每个数组（batch，21，1）
        return encoder_inputs, _extras, _labels

    #第一层lstm
    def EncoderLSTM(self, encoder_cell1, encoder_cell2, encoder_cell3, encoder_cell4, encoder_cell5, encoder_cell6, encoder_cell7, encoder_cell8, encoder_cell9, encoder_cell10,
                    encoder_cell11, encoder_cell12, encoder_inputs, extra, DisM, AngleM, encoder_target):
        Others = encoder_inputs #12 (batch, 11, 28)
        batch_size = Others[0].data.size(0)
        OthersTensor = Others[0].float()
        extraTensor = extra[0].float()
        DisM = np.array(DisM)
        AngleM = np.array(AngleM)
        DisMTensor = torch.from_numpy(DisM).float()
        DisMTensor = DisMTensor.repeat(batch_size, 1, 1)
        AngleMTensor = torch.from_numpy(AngleM).repeat(batch_size, 1, 1).squeeze() #(batch_size,12)
        alist = []
        for i in range(12):
            ci = np.exp(-(DisM[i] ** 2)/(2 * (self.sigma.detach().numpy()** 2)))
            Pearsonij = pearsonr(OthersTensor[0, 8:, i], extraTensor[0, 5:, 0])
            si = np.exp(-(abs(Pearsonij[0]) ** 2)/(2 * (self.sigmas.detach().numpy()** 2)))
            alist.append(ci*si)
        alist = np.array(alist)
        alistTensor = torch.FloatTensor(alist)
        weight_a = torch.softmax(alistTensor, dim=0)  # A of first layer 11,1
        weight_aa = weight_a.reshape(1, 1, -1).repeat(batch_size, 100, 1)
        predictlist=[]
        index = 0
        for flinput, ex in zip(Others, extra): #n个时间步
            s1 = flinput[:, 0:8, 0].float()
            s2 = flinput[:, 0:8, 1].float()
            s3 = flinput[:, 0:8, 2].float()
            s4 = flinput[:, 0:8, 3].float()
            s5 = flinput[:, 0:8, 4].float()
            s6 = flinput[:, 0:8, 5].float()
            s7 = flinput[:, 0:8, 6].float()
            s8 = flinput[:, 0:8, 7].float()
            s9 = flinput[:, 0:8, 8].float()
            s10 = flinput[:, 0:8, 9].float()
            s11 = flinput[:, 0:8, 10].float()
            s12 = flinput[:, 0:8, 11].float()
            target = ex[:, 0:5, 0].float()
            htarget, ctarget = encoder_target(target)
            h1, c1 = encoder_cell1(s1)
            h2, c2 = encoder_cell2(s2)
            h3, c3 = encoder_cell3(s3)
            h4, c4 = encoder_cell4(s4)
            h5, c5 = encoder_cell5(s5)
            h6, c6 = encoder_cell6(s6)
            h7, c7 = encoder_cell7(s7)
            h8, c8 = encoder_cell8(s8)
            h9, c9 = encoder_cell9(s9)
            h10, c10 = encoder_cell10(s10)
            h11, c11 = encoder_cell11(s11)
            h12, c12 = encoder_cell12(s12)
            con = torch.stack([h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12], dim=2)
            alist = []
            for i in range(12):
#                if index !=0:
#                    con1 = torch.cat([con, lastcon], dim=1)
#                    #con1 = self.b *con +(1-self.b)*lastcon
#                    con1 = torch.matmul(self.wp, con1)+self.bp
#                    htarget1 =  torch.cat([htarget.reshape(batch_size, -1, 1), lasttarget], dim=1)
#                    htarget1 = torch.matmul(self.wp, htarget1) +self.bp
#                else:
#                    con1 = con
#                    htarget1 = htarget        
                indexT = con[:, :, i]
                cat = torch.cat([htarget, indexT], dim=1)
                a = torch.relu(torch.matmul(cat, self.Attention1)+self.biasa1)
                wij = torch.cat([flinput[:, 7, i:i + 1], flinput[:, 6, i:i + 1]], dim=1).float()
                fc = torch.cat([a, wij], dim=1)   #batch_size,102
                b = torch.relu(torch.matmul(fc, self.Attention2)+self.biasa2)              
                alist.append(b.tolist())
            #lastcon = con
            #lasttarget = htarget.reshape(batch_size, -1, 1)   
            alist = np.squeeze(np.array(alist))
            alistTensor = torch.FloatTensor(alist)
            weight_a = torch.softmax(alistTensor, dim=0)
            weight_a = weight_a.reshape(batch_size, 1, 12).repeat(1, 300, 1)  #batch_size,300,11    
            #flinput = torch.relu(torch.matmul(self.ff, flinput[:, 8:, :].float())+self.bff)
            #fusiondis = torch.sum(flinput * weight_aa, dim=2).float()  
            Fusion_result = torch.sum(con * weight_a, dim=2)  #batch 300
            Fusion_result = torch.cat([Fusion_result, htarget], dim=1)
            fuse = torch.matmul(Fusion_result, self.fuse2) +self.biasf2
            #fuseall = torch.cat([fuse, fusiondis], dim=1)
            #predict = torch.matmul(self.a * fuse +(1-self.a)*fusiondis, self.Wout)+self.biasout
            predict = torch.matmul(fuse, self.Wout) + self.biasout
            predictlist.append(predict)
            index = index+1
        return predictlist
    def forward(self, x, DisM, AngleM):
        encoder_inputs, extras, labels = self.input_transform(x)
        outputs = self.EncoderLSTM(self.encoder_cell1, self.encoder_cell2, self.encoder_cell3, self.encoder_cell4,
                                                                                  self.encoder_cell5, self.encoder_cell6, self.encoder_cell7, self.encoder_cell8,
                                                                                  self.encoder_cell9, self.encoder_cell10, self.encoder_cell11, self.encoder_cell12, encoder_inputs, extras, DisM, AngleM, self.encoder_target)
        return outputs, labels
