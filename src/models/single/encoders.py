import torch
from torch import nn, Tensor

from .base_encoders import Base_Encoder

class MLP(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_sizes: tuple = None,
        activation=nn.ReLU, #LeakyReLU, GELU or nn.Tanh()
        dropout=0,
        batchnorm: bool=False,
        **kwargs,
    ):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes
        self.encoder_output = layer_sizes[-1]
        layers = []
        # other layers
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
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

        if self.unit_type == "gru":
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
        rnn_out, (h_n, c_n) = self.rnn(x)
        rnn_out = rnn_out[:, -1] # only consider output of last time step-- what about attention-aggregation
        return {"rep": self.fc(rnn_out)}

    def get_output_size(self):
        return self.layer_size

class TransformerNet(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_size: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        num_heads: int = 1,
        len_max_seq: int = 24,
        fixed_pos_encoding: bool = True,
        pre_embedding=True,
        **kwargs,
    ):
        super(TransformerNet, self).__init__()
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.len_max_seq = len_max_seq+ 1 #for class token
        self.num_heads = num_heads
        self.fixed_pos_encoding = fixed_pos_encoding

        self.embedding_layer = EmbeddingLayer(in_features=self.feature_size, out_features=self.layer_size, model_type="linear") if pre_embedding else nn.Identity()
        self.emb_dim = self.layer_size if pre_embedding else self.feature_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))  
        self.pos_encoder = PositionalEncoding(
            d_model=self.emb_dim, 
            dropout=self.dropout,
            max_len=self.len_max_seq)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.num_heads,
            #dim_feedforward=self.layer_size,
            dropout=self.dropout,
            batch_first=True,
            **kwargs
        )
        self.tr_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=self.num_layers)

    def forward(self, x, src_key_padding_mask=None, **kwargs):
        x = self.embedding_layer(x)
        # add cls token to input sequence
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1)

        # add time step for cls token in src_key_padding_mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat(
                (
                    torch.zeros_like(src_key_padding_mask[:, -1].reshape(-1, 1)).bool(),  #always false to attend cls token
                    src_key_padding_mask,  # for rest fo x
                ),
                dim=1)
        if self.fixed_pos_encoding: # add position encoding
            x = self.pos_encoder(x)
        
        # pass x through transformer encoder
        x = self.tr_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return {"rep": x[:, 0, :]} # extract class token feature: 

    def get_output_size(self):
        return self.emb_dim

class PositionalEncoding(nn.Module):
    """
    ref: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding

    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # create a long enough position encoding
        position_encoding = torch.zeros(1, max_len, d_model, )

        position = torch.arange(
            max_len, dtype=torch.float32
        ).reshape(-1, 1) / torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        position_encoding[0, :, 0::2] = torch.sin(position)
        position_encoding[0, :, 1::2] = torch.cos(position)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("position_encoding", position_encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.position_encoding[:, :x.shape[1], :]  # based on the seq len of x, add position encoding
        return self.dropout(x)
    

class EmbeddingLayer(nn.Module):
    def __init__(self, in_features, out_features, model_type="linear", norm_layer=False):
        """
        Note: model_type: linear and conv1d with kernel size gives same result, linear is bit faster
            ref: https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer
        Parameters
        ----------
        in_features:
        out_features
        model_type
        """
        super().__init__()
        self.model_type = model_type
        if self.model_type == "linear":
            self.model = nn.Linear(in_features=in_features, out_features=out_features)
        elif self.model_type == "conv1d":
            self.model = nn.Conv1d(  # for conv 1d req shape: batch_size x in_features x seq_len
                in_features, out_features, 1
            )
        else:
            raise KeyError(f"Model type: {model_type} is not implemented for {self._get_name()} class")
        self.norm_layer = nn.LayerNorm(out_features) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.model_type == "linear":
            x= self.model(x)
        elif self.model_type == "conv1d":
            x = self.model(x.permute(0, 2, 1)).permute(  # for conv 1d req shape: batch_size x in_features x seq_len
                0, 2, 1
            )  # we change output shape back to shape: batch_size x seq_len x in_features
        return self.norm_layer(x)