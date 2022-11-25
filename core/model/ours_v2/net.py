# --------------------------------------------------------
# hrvqa
# Licensed under The MIT License [see LICENSE for details]
# Written by lik
# --------------------------------------------------------

from core.model.mca_gated.ga1 import GA_ED
from core.model.net_utils import FC, MLP, LayerNorm, MLP2

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class AttFlat_fusion(nn.Module):
    def __init__(self, __C):
        super(AttFlat_fusion, self).__init__()
        self.__C = __C

        self.mlp1 = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.mlp2 = MLP2(
            in_size=__C.HIDDEN_SIZE * 2,
            mid_size=__C.FLAT_MLP_SIZE * 2,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask, y, y_mask):
        att_x = self.mlp1(x)
        att_x = att_x.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att_x = F.softmax(att_x, dim=1)

        att_y = self.mlp1(y)
        att_y = att_y.masked_fill(
            y_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att_y = F.softmax(att_y, dim=1)


        att_list_x = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list_x.append(
                torch.sum(att_x[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list_x, dim=1)
        x_atted = self.linear_merge(x_atted)

        att_list_y = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list_y.append(
                torch.sum(att_y[:, :, i: i + 1] * y, dim=1)
            )

        y_atted = torch.cat(att_list_y, dim=1)
        y_atted = self.linear_merge(y_atted)


        fusion_att = x_atted + y_atted

        att_fusion = self.mlp2(fusion_att)

        att_fusion = F.softmax(att_fusion, dim=1)

        x_atted1 = att_fusion[:, :] * x_atted

        y_atted1 = att_fusion[:, :] * y_atted


        return x_atted1, y_atted1

# -------------------------
# ---- Main HRVQA Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        # self.embedding = nn.Embedding(
        #     num_embeddings=token_size,
        #     embedding_dim=__C.WORD_EMBED_SIZE
        # )

        # # Loading the GloVe embedding weights
        # if __C.USE_GLOVE:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # load bert
        self.bert = BertModel.from_pretrained(self.__C.PRETRAINED_PATH)
        
        self.lstm = nn.LSTM(
            input_size=self.__C.PRETRAINED_HIDDEN,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # self.lstm = nn.LSTM(
        #     input_size=__C.WORD_EMBED_SIZE,
        #     hidden_size=__C.HIDDEN_SIZE,
        #     num_layers=1,
        #     batch_first=True
        # )

        # self.adapter = Adapter(__C)
        self.img_feat_linear = nn.Linear(__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE)

        self.backbone = GA_ED(__C)
        

        self.attflat = AttFlat_fusion(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_idx, attention_mask,token_type_ids):

        # Pre-process Language Feature
        lang_feat_mask = self.make_mask(ques_idx.unsqueeze(2))
        # lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.bert(input_ids=ques_idx, attention_mask=attention_mask, token_type_ids=token_type_ids)
        lang_feat, _ = self.lstm(lang_feat)

        # img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        img_feat_mask = self.make_mask(img_feat)
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )


        img_feat, lang_feat =  self.attflat(img_feat, img_feat_mask, lang_feat, lang_feat_mask)

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat
    

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

