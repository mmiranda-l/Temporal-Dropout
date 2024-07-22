import torch
from torch import nn

from src.models.single.base_encoders import Base_Encoder, Generic_Encoder
from src.layers.dropconnect import WeightDropLinear, WeightDropGRU, WeightDropLSTM
from src.layers.concrete_temporal_dropout import ConcreteTemporalDropout
from src.layers.temporal_dropout import TemporalDropout
class MLP(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_sizes: tuple = None,
        activation=nn.ReLU, #LeakyReLU, GELU or nn.Tanh()
        dropout=0,
        batchnorm: bool=False,
        approx_type: str=None,
        **kwargs,
    ):
        super(MLP, self).__init__()
        approx_type = approx_type
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes
        self.encoder_output = layer_sizes[-1]
        layers = []
        # other layers
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]) if approx_type != "dropconnect" else WeightDropLinear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation(),
                    nn.BatchNorm1d(layer_sizes[l_id+1], affine=True) if batchnorm else nn.Identity(),
                    nn.Dropout(p=dropout) if dropout!=0 else nn.Identity(),
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return {"rep": self.layers(x)}

    def get_output_size(self):
        return self.encoder_output

class RNNet(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_size: int = 128,
        dropout: float =0,
        num_layers: int = 1,
        bidirectional: bool = False,
        unit_type: str="gru",
        batchnorm: bool = False,
        temporal_pool: bool = False,
        pack_seq: bool = False,
        **kwargs,
    ):
        super(RNNet, self).__init__()
        self.unit_type = unit_type.lower()
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batchnorm = batchnorm
        self.temporal_pool = temporal_pool
        self.pack_seq = pack_seq

        if self.unit_type in "gru":
            rnn_type_class = torch.nn.GRU
        elif self.unit_type == "lstm":
            rnn_type_class = torch.nn.LSTM
        elif self.unit_type == "rnn":
            rnn_type_class = torch.nn.RNN
        else:
            pass #raise error

        self.rnn = rnn_type_class(
                input_size=self.feature_size,
                hidden_size=self.layer_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)

        self.fc = torch.nn.Sequential(
            nn.BatchNorm1d(self.layer_size) if self.batchnorm else nn.Identity(),
        )


    def forward(self, x):
        # if self.pack_seq:
        #     lengths = torch.Tensor([ (torch.isfinite( v ).sum(axis=-1)!=0).sum() for v in x]).cpu() 
        #     x =  nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        rnn_out, (h_n, c_n) = self.rnn(x)

        # if self.pack_seq:
        #     seq_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, padding_value=0.0 )
        #     masks = (lengths-1).view(-1, 1, 1).expand(-1, seq_unpacked.size(1), seq_unpacked.size(2)).to(seq_unpacked.device).type(torch.int64)
        #     rnn_out = seq_unpacked.gather(0, masks)

        rnn_out = rnn_out[:, -1] # only consider output of last time step-- what about attention-aggregation
        return {"rep": self.fc(rnn_out)}

    def get_output_size(self):
        return self.layer_size

class BayesianEncoders(Base_Encoder):
    def __init__(self, 
        feature_size: int,
        layer_size: int = 128,
        dropout: float =0.3,
        num_layers: int = 2,
        bidirectional: bool = False,
        network_type: str="lstm",
        approx_type: str="dropout",
        batchnorm: bool = False,
        weight_regularizer: float=1e-4,
        dropout_regularizer: float=2e-4, 
        concrete_td: bool= True,
        **kwargs,
                 ) -> None:
        super(BayesianEncoders, self).__init__()

        self.approx_type = approx_type.lower()
        self.network_type = network_type.lower()
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batchnorm = batchnorm

        if self.approx_type == "dropout" and self.network_type == "gru":
            rnn_type_class = torch.nn.GRU
        elif self.approx_type == "dropout" and self.network_type == "lstm":
            rnn_type_class = torch.nn.LSTM
        elif self.approx_type == "dropconnect" and self.network_type == "lstm":
            rnn_type_class = WeightDropLSTM
        elif self.approx_type == "dropconnect" and self.network_type == "gru":
            rnn_type_class = WeightDropGRU

        
        #self.concrete_td = ConcreteTemporalDropout(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer) if concrete_td else nn.Identity()

        self.rnn = rnn_type_class(
            input_size=self.feature_size,
            hidden_size=self.layer_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        self.fc = torch.nn.Sequential(
            nn.BatchNorm1d(self.layer_size) if self.batchnorm else nn.Identity(),
        )
    
    def forward(self, x):

       # x, _ = self.concrete_td(x)
        rnn_out, states = self.rnn(x)
        rnn_out = rnn_out[:, -1] # only consider output of last time step-- what about attention-aggregation
        return {"rep": self.fc(rnn_out)}

    def get_output_size(self):
        return self.layer_size
    

    

if __name__ == "__main__":
    b = BayesianEncoders(
        feature_size=2,
        network_type="lstm",
        approx_type="dropout", 

    )
    print(b)
    t = torch.ones(5, 3, 2)
    print(t.shape)
    out = b(t)
    print(out["rep"].shape)
