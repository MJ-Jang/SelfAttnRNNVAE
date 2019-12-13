import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable




class BiLSTMEncoder(nn.Module):

    def __init__(self, vocab: int, emb_size: int, hidden_dim: int,
                 nlayers: int, dropout: float, pad_id: int):
        super(BiLSTMEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_dim, nlayers, dropout=dropout,
                              bidirectional=True)
        self.nlayers = nlayers
        self.nhid = hidden_dim
        self.pad_id = pad_id

        self.encoder.weight.data[self.pad_id] = 0

    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp):
        hidden = [hid.to(inp.device) for hid in self.init_hidden(inp.size(0))]
        emb = self.drop(self.encoder(inp.transpose(0,1)))
        outp = self.bilstm(emb, hidden)[0]
        outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, vocab: int, emb_size: int, hidden_dim: int,
                 nlayers: int, attn_unit: int, attn_hops: int,
                 dropout: float, pad_id: int):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTMEncoder(vocab, emb_size, hidden_dim,
                                    nlayers, dropout, pad_id)
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(hidden_dim * 2, attn_unit, bias=False)
        self.ws2 = nn.Linear(attn_unit, attn_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.attention_hops = attn_hops
        self.pad_id = pad_id

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp):
        outp = self.bilstm(inp)[0]
        size = outp.size()
        # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])
        # [bsz*len, nhid*2]d
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()
        # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])
        # [bsz, 1, len]
        concatenated_inp = [transformed_inp for _ in
                            range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)
        # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))
        # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)
        # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()
        # [bsz, hop, len]
        penalized_alphas = alphas + (
                -10000 * (concatenated_inp == self.pad_id).float())
        # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))
        # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])
        # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)


class SelfAttnVariationalEncoder(nn.Module):
    def __init__(self, vocab: int, hidden_dim: int, latent_dim:int,
                 nlayers: int, attn_unit: int, attn_hops: int,
                 dropout: float, pad_id: int):
        super(SelfAttnVariationalEncoder, self).__init__()
        self.enc = SelfAttentiveEncoder(vocab=vocab,
                                        emb_size=hidden_dim,
                                        hidden_dim=hidden_dim,
                                        nlayers=nlayers,
                                        attn_unit=attn_unit,
                                        attn_hops=attn_hops,
                                        dropout=dropout,
                                        pad_id=pad_id)

        self.mu_enc = nn.Linear(hidden_dim * 2, latent_dim)
        self.log_sigma_enc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, input_ids: torch.LongTensor, is_train: bool = True):
        output, alphas = self.enc(input_ids)
        final_state = output.mean(1)
        return self._sample_latent(final_state, is_train)

    def _sample_latent(self, h_enc, is_train):
        mu = self.mu_enc(h_enc)
        if not is_train:
            return mu
        else:
            log_sigma = self.log_sigma_enc(h_enc)
            sigma = torch.exp(log_sigma)
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
            latent = mu + sigma * Variable(std_z, requires_grad=False).to(h_enc.device)
            return latent, mu, sigma


class SelfAttnVAE(nn.Module):
    def __init__(self,
                 vocab: int,
                 hidden_dim: int,
                 latent_dim: int,
                 nlayers: int,
                 attn_unit: int,
                 attn_hops:int,
                 dropout: float,
                 pad_id: int):
        super(SelfAttnVAE, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = SelfAttnVariationalEncoder(vocab=vocab,
                                                  hidden_dim=hidden_dim,
                                                  latent_dim=latent_dim,
                                                  nlayers=nlayers,
                                                  attn_unit=attn_unit,
                                                  attn_hops=attn_hops,
                                                  dropout=dropout,
                                                  pad_id=pad_id)

        self.word_embeddings = self.encoder.enc.bilstm.encoder

        self.dec_gru = nn.GRU(hidden_dim, latent_dim, num_layers=1)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(latent_dim, vocab)

    def forward(self,
               enc_input_ids: torch.LongTensor,
               dec_input_ids: torch.LongTensor,
               dec_length: torch.LongTensor):

        z, mu, sigma = self.encoder(enc_input_ids)
        logits, dec_max_len = self.decode(z, dec_input_ids, dec_length)
        return logits, dec_max_len, mu, sigma

    def decode(self,
               z,
               input_ids: torch.LongTensor,
               length: torch.LongTensor):
        input_vectors = self.word_embeddings(input_ids)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_vectors,
                                                               length.tolist(),
                                                               batch_first=True,
                                                               enforce_sorted=False)
        output, _ = self.dec_gru(packed_input, z.unsqueeze(0))
        output, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output, int(max(out_lengths))
