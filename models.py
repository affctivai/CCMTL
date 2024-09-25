from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modulator import Modulator
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

import torch.nn.functional as F
import torch.nn as nn
import torch

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""
    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/utils.py
def normalize_A(A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/utils.py
def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/layers.py
class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/layers.py
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class DGCNN(nn.Module):
    def __init__(self, args):
        #in_channels(int): The feature dimension of each electrode.

        if args.data_choice =='dreamer':
            in_channels=3
            num_electrodes=14
        elif args.data_choice == 'deap':    
            in_channels = 5
            num_electrodes = 32
        else:
            in_channels= 5
            num_electrodes=62
                
        k_adj = 2
        num_classes=args.n_classes
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        out_channels=256
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        x=torch.squeeze(x, axis=1)

       
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline

        L = normalize_A(self.A)

        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)

        return result


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCMTL(nn.Module):
    def __init__(self, args):
        super(CCMTL, self).__init__()

        self.args=args
        lstm_hidden_size=args.lstm_hidden_size       
        args.gate_channels=64
        n_units=args.n_units
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
        )

        self.fc_layer=nn.Sequential(
            Flatten(),
            nn.Linear(4864,n_units)            
        )


        encoder_layer = nn.TransformerEncoderLayer(d_model=n_units, nhead=2)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer, num_layers=1)

        rnn = nn.LSTM

        self.eeg_rnn1 = rnn(n_units, int(lstm_hidden_size), bidirectional = True)
        self.eeg_rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional = True)
        
        if args.lstm:
            fc_in_features=4*int(lstm_hidden_size)
        else:
            fc_in_features=n_units

        self.fc = nn.Linear(in_features=fc_in_features, out_features= args.n_classes)
        self.modulator = Modulator(args)

    def convNet(self, x):
        o= self.cnn(x)

        return o

    def sLSTM(self, x, lengths):
        batch_size = lengths.size(0)
        packed_h1, (final_h1, _) = self.eeg_rnn1(x)
        _, (final_h2, _) = self.eeg_rnn2(final_h1)

        o = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        o = self.fc(o)

        return o

    def forward(self, x):

        o=self.convNet(x)
       
        if self.args.modulator:    
            o=self.modulator(o)

        lengths = torch.LongTensor([x.shape[1]]*x.size(0))

        o=self.fc_layer(o)

          
        o=torch.unsqueeze(o, dim=0)
        o=self.transformer_encoder(o)
             
        if self.args.lstm:        
            o = self.sLSTM(o, lengths)
        else:
            o=torch.squeeze(o,axis=0)            
            o=self.fc(o)
        
         
        return o



class MT_CNN(nn.Module):
    def __init__(self, args):
        super(MT_CNN, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            Flatten(),
            nn.Linear(9920,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )

        self.fc = nn.Linear(in_features=512, out_features= args.n_classes)

    def forward(self, x):
        x=self.cnn_layer(x)
        x=self.fc(x)

        return x

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name)  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.selu(tf.add(conv2d(x, weight, kernel_stride),bias))

class CCNN(nn.Module):
    def __init__(self,args):
        super(CCNN, self).__init__()

        self.conv1=nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=1, padding='same')

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=1,padding='same')

        self.conv3 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size=4, stride=1, padding='same')

        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding='same')

        self.fc = nn.Sequential()
        self.fc.add_module('fc_layer1', nn.Linear(in_features=19840, out_features=1024))
        self.fc.add_module('fc_layer1_dropout', nn.Dropout(0.2))
        self.fc.add_module('fc_layer2', nn.Linear(in_features=1024, out_features=args.n_classes))

    def forward(self, x):
        # print('x:', x.shape)
        # exit()
        # x=x.permute(1,0,2).contiguous()
        batch_size=x.size(0)
        x = self.conv1(x)
        x=F.elu(x)
        x = self.conv2(x)
        x=F.elu(x)
        x = self.conv3(x)
        x=F.elu(x)
        x = self.conv3_2(x)
        x=F.elu(x)
        x=x.view(batch_size,-1)

        x = self.fc(x)

        return x

class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4*16*122, args.n_classes)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
       
        x = self.conv1(x)

        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)

        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)

        x = self.conv2(x)

        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        # x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)

        # x = self.pooling3(x)

        # FC Layer

        x = x.reshape(-1, 4*16*122)

        x = self.fc1(x)

        return x


class PCRNN(nn.Module):
    def __init__(self,args):
        super(PCRNN, self).__init__()
        emb_size = 310
        hidden_size = 512
        in_features=19840

        self.conv1=nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=1, padding='same')

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=1,padding='same')

        self.conv3 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size=4, stride=1, padding='same')

        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding='same')

        self.fc = nn.Sequential()
        self.fc.add_module('fc_layer1', nn.Linear(in_features=in_features, out_features=1024))
        self.fc.add_module('fc_layer1_dropout', nn.Dropout(0.5))

        rnn = nn.LSTM

        self.eeg_rnn1 = rnn(emb_size, int(hidden_size), bidirectional = False)
        self.eeg_rnn2 = rnn(hidden_size, int(hidden_size), bidirectional = False)
        fc_input_size = 4*int(hidden_size)

        self.fusion_layer = nn.Linear(in_features = 2048, out_features = args.n_classes)

    def CNN(self, x):
        batch_size=x.size(0)
        x = self.conv1(x)
        x=F.elu(x)
        x = self.conv2(x)
        x=F.elu(x)
        x = self.conv3(x)
        x=F.elu(x)
        x = self.conv3_2(x)
        x=F.elu(x)
        x=x.view(batch_size,-1)

        x = self.fc(x)

        return x

    def sLSTM(self, x, lengths):
        x= x.permute(1,0,2)

        batch_size = lengths.size(0)
        lengths = lengths.detach().cpu().numpy()
        #p_seq = pack_padded_sequence(x, lengths)
        packed_out1, (final_out1, _) = self.eeg_rnn1(x)

        _, (final_out2, _) = self.eeg_rnn2(final_out1)
        
        o = torch.cat((final_out1, final_out2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        return o

    def forward(self, x):
        cnn_output = self.CNN(x)
        lengths = torch.LongTensor([x.shape[1]]*x.size(0))
        lstm_output = self.sLSTM(x, lengths)
        fusion = torch.cat((cnn_output, lstm_output), dim=1).contiguous().view(lengths.size(0), -1)

        o = self.fusion_layer(fusion)

        return o

class CNNTransformer(nn.Module):
    def __init__(self, args):
        super(CNNTransformer, self).__init__()

        self.args=args
        f_units = args.f_units

        self.layer_maxpool = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.AvgPool1d(4,2),
            Flatten(),
            nn.Linear(4736,f_units)
        )

        _input_size=int(args._input_size)

        if args.modulator:
            self.modulator = Modulator(f_units, args)
        encoder_layer = nn.TransformerEncoderLayer(d_model=f_units, nhead=2)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(in_features=f_units, out_features= args.n_classes)

    def convNet(self, x):
        o=x_maxpool = self.layer_maxpool(x)

        return o

    def forward(self, x):
        o=self.convNet(x)

        o=torch.unsqueeze(o, dim=0)
        o=self.transformer_encoder(o)
        o=torch.squeeze(o, dim=0)

        o = self.fc(o)

        return o

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            Flatten(),
            nn.Linear(4864,emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.shallownet(x)
        x= torch.unsqueeze(x, axis=1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)

        return x, out

class Conformer(nn.Sequential):
    def __init__(self, args, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, args.n_classes)
        )

#
