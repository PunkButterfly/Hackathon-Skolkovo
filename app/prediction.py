import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


def embed_bert_cls(text, bert_tokenizer, bert_model):
    t = bert_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = bert_model(**{k: v.to(bert_model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


class FullyNet(torch.nn.Module):
    def __init__(self, input_dim, out_dim=1):
        super(FullyNet, self).__init__()
        self.fc_1 = torch.nn.Linear(input_dim, 384)
        self.fc_2 = torch.nn.Linear(384, 256)
        self.fc_3 = torch.nn.Linear(256, 128)
        self.fc_4 = torch.nn.Linear(128, 64)
        self.fc_5 = torch.nn.Linear(64, 8)
        self.fc_6 = torch.nn.Linear(8, out_dim)

    def forward(self, x):
        x = self.fc_1(F.normalize(x))
        x = self.fc_2(F.normalize(x))
        x = self.fc_3(F.normalize(x))
        x = self.fc_4(F.normalize(x))
        x = self.fc_5(F.normalize(x))
        out = self.fc_6(F.normalize(x))
        return out


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.input_size = inputSize
        self.output_size = outputSize
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def make_predict(model, data):
    model.eval()
    return model(data).item()


def df_to_embed_tensor(ser, bert_tokenizer, bert_model):
    x_candidates = embed_bert_cls(ser['Candidate_descr'], bert_tokenizer, bert_model)
    x_job = embed_bert_cls(ser['Jobs_descr'], bert_tokenizer, bert_model)

    x = np.concatenate((x_candidates, x_job))
    return torch.Tensor(x).view(1, -1)


def inference(weights_path, data, bert_tokenizer, bert_model):
    """
        In:
            weights_path - path to model weights .pt
            data - as Series
            ModelClass - class of model (default FullyNet)
    """
    input_dim = 312 * 2
    out_dim = 1

    embed = df_to_embed_tensor(data, bert_tokenizer, bert_model)

    model = FullyNet(input_dim, out_dim)

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    return make_predict(model, embed)


# path = '/content/fully_model.pt'
# df = data.loc[0, :].to_frame().T
# inference(path, df, FullyNet)
