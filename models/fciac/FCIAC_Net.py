import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
from models.resnet18_encoder import resnet18
from models.resnet20_cifar import resnet20

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden_ch = int(in_ch * expand)
        self.use_residual = stride == 1 and in_ch == out_ch
        layers = []

        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.SiLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=stride, padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

## The lightweight plastic embedding extractor
class StudentMobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_features=128, width_mult=0.5, dropout=0.1):
        super().__init__()

        def c(ch):  # 通道数缩放
            return max(8, int(ch * width_mult))

        # 初始stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c(16), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)),
            nn.SiLU(inplace=True),
        )

        # 配置：[expand_ratio, output_channel, repeats, stride]
        inverted_residual_cfg = [
            (2, c(24), 2, 2),  # 两层，下采样一次
            (2, c(32), 2, 2),  # 两层，下采样一次
            (2, c(64), 2, 2),  # 两层，下采样一次
            (2, c(96), 1, 1),  # 一层，不下采样
        ]

        layers = []
        in_ch = c(16)
        for expand_ratio, output_ch, repeats, stride in inverted_residual_cfg:
            for i in range(repeats):
                s = stride if i == 0 else 1
                layers.append(InvertedResidual(in_ch, output_ch, s, expand_ratio))
                in_ch = output_ch

        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_features),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x
# 融合模块（自适应门控）
class FusionModule(nn.Module):
    def __init__(self, teacher_dim=512, student_dim=128, fused_dim=512):
        super().__init__()
        #self.teacher_proj = nn.Linear(teacher_dim, fused_dim)
        self.student_proj = nn.Linear(student_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(2*fused_dim, fused_dim//2),
            nn.ReLU(),
            nn.Linear(fused_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_teacher, feat_student,return_alpha:bool = False):
        teacher_feat_proj = feat_teacher
        student_feat_proj = self.student_proj(feat_student)
        gate_input = torch.cat([teacher_feat_proj, student_feat_proj], dim=-1)
        alpha_ = self.gate(gate_input)
        alpha = 0.6 + 0.4 * alpha_   #LS 0.6    #FS 0.7
        fused_feat = alpha * teacher_feat_proj + (1 - alpha) * student_feat_proj
        if return_alpha:
            return fused_feat, alpha_
        return fused_feat   
    
class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)
        self.session = 0  # 初始化 session 属性
        self.mode = mode
        self.args = args
        self.encoder = resnet18(True, args)  # pretrained=False SEE
        self.num_features = 512
        self.fc = nn.Linear(self.num_features, self.args.num_all, bias=False)
        hdim=self.num_features
        self.beta = 1.0
        self.student_model = StudentMobileNetV2(in_channels=3, num_features=128, width_mult=0.5, dropout=0.1)#LPEE

        # 融合模块
        self.fusion_module = FusionModule(
        teacher_dim=self.num_features, # 512
        student_dim=128,               # 学生模型输出维度
        fused_dim=self.num_features    # 保持512
        )

        # 门控融合网络：替代静态 β
        self.gate_fn = nn.Sequential(
            nn.Linear(2 * self.num_features, self.num_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_features // 2, 1),
            nn.Sigmoid()
        )
        #CPAN
        # 生成模块
        self.transatt_proto = MultiHeadAttention(1, self.num_features, self.num_features, self.num_features, dropout=0.5)
        #for p in self.transatt_proto.parameters(): p.requires_grad = False
        # 增量原型分支，仅在增量阶段训练
        self.inc_attn = MultiHeadAttention(1, self.num_features, self.num_features, self.num_features, dropout=0.5)
       

        # 自适应层
        self.slf_attn = MultiHeadAttention(1, self.num_features, self.num_features, self.num_features, dropout=0.5)


        if args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
            self.set_fea_extractor_for_s2s()
        else:
            self.set_module_for_audio(args)

    def set_session(self, session):
        """设置当前会话编号，在训练循环中调用"""
        self.session = session
            
    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        elif self.mode == 'fm_encoder':
            if self.args.dataset == "fsdclips":
                x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
                x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                # x = x.transpose(1, 3)
                # x = self.bn0(x)
                # x = x.transpose(1, 3)
                x = x.repeat(1, 3, 1, 1)
            input = self.encoder(input)
            return input
        elif self.mode == 'fusion_encoder': # 新增融合模式

            teacher_feat = self.encode(input)        # 教师模型特征
            input=self.waveform_to_img(input)
            student_feat = self.student_model(input) # 学生模型特征
            fused_feat = self.fusion_module(teacher_feat, student_feat) # 融合
            return fused_feat
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)
            return logits

    def _forward(self, support, query, pqa=False, sup_emb=None, novel_ids=None):  # support and query are 4-d tensor, shape(num_batch, 1, num_proto, emb_dim)
        anchor_loss = 0.0
        emb_dim = support.size(-1)
        num_query = query.shape[1]*query.shape[2]#num of query*way
        query = query.view(-1, emb_dim).unsqueeze(1)  # shape(num_query, 1, emb_dim)

         # get mean of the support of shape(batch_size, shot, way, dim)
        mean_proto = support.mean(dim=1, keepdim=True)  # calculate the mean of each class's prototype without keeping the dim

        num_batch = mean_proto.shape[0]
        num_proto = mean_proto.shape[2]  # num_proto = num of support class

        # the shape of proto is different from query, so make them same by coping (num_proto, emb_dim)
        mean_proto_expand = mean_proto.expand(num_batch, num_query, num_proto, emb_dim).contiguous()  # can be regard as copying num_query(int) prot
       

        if sup_emb is not None:
            att_proto = self.get_att_proto(sup_emb, query, num_query, emb_dim)
            mean_proto_expand.data[:, :, novel_ids, :] = self.beta * att_proto.unsqueeze(0) \
                                                    + (1-self.beta) * mean_proto_expand[:, :, novel_ids, :]
        proto = mean_proto_expand.view(num_batch*num_query, num_proto, emb_dim)

        if pqa:
            combined = torch.cat([proto, query], 1)  # Nk x (N + 1) or (N + 1 + 1) x d, batch_size = NK
            if self.session > 0:
               #combined1, _ = self.slf_mamba(combined)
               combined1, _ = self.slf_attn(combined, combined, combined)

               
               #combined2,_ = self.inc_block(combined)
               combined2,_ =self.inc_attn(combined, combined, combined)
               #combined2=combined2.view(num_batch * num_query, num_proto + 1, emb_dim)
               gate_input = torch.cat([combined1, combined2], dim=-1)  # (bs, way, 2*dim)
               gate = self.gate_fn(gate_input)                           # (bs, way, 1)
               combined = (1 - gate) * combined1 + gate * combined2 
               proto, query = combined.split(num_proto, dim=1)
            else:
               #combined, _ =self.slf_mamba(combined)
               combined1=None
               combined2=None
               combined, _ = self.slf_attn(combined, combined, combined)
               proto, query = combined.split(num_proto, dim=1)
        else:
            combined = proto
            combined, _ = self.slf_attn(combined, combined, combined)
            proto = combined


        logits=F.cosine_similarity(query, proto, dim=-1)
        logits=logits*self.args.network.temperature
        return logits, anchor_loss, query, proto,combined1 if pqa else None, combined2 if pqa else None
    
     
    
    def get_att_proto_shot_score(self, sup_emb, num_query, emb_dim):
        sup_emb = sup_emb.view(self.args.episode.episode_shot, -1, emb_dim).permute(1, 0, 2)
        att_emb, att_logit = self.inneratt_proto(sup_emb, sup_emb, sup_emb)
        # att_proto = att_emb.mean(dim=1)

        shot_logit = att_logit.mean(dim=1)
        shot_score = F.softmax(shot_logit, dim=1)
        shot_score = shot_score.unsqueeze(-1)
        att_proto = shot_score * sup_emb
        att_proto = att_proto.sum(1)
        att_proto_expand = att_proto.unsqueeze(0).expand(num_query, -1, emb_dim)
        return att_proto_expand

    def get_att_proto(self, sup_emb, query, num_query, emb_dim):
        sup_emb = sup_emb.unsqueeze(0).expand(num_query, sup_emb.shape[0], sup_emb.shape[-1])
        cat_emb = torch.cat([sup_emb, query], dim=1)
        att_pq, att_logit = self.transatt_proto(cat_emb, cat_emb, cat_emb)
        att_logit = att_logit[:, :, -1][:, :-1] # 选取最后一列的前shot*way个logits
        att_score = torch.softmax(att_logit.view(num_query, self.args.episode.episode_shot, -1), dim=1)
        att_proto, _ = att_pq.split(sup_emb.shape[1], dim=1)
        att_proto = att_proto.view(num_query, self.args.episode.episode_shot, -1, emb_dim) * att_score.unsqueeze(-1) # self.args.episode_way+self.args.low_way
        att_proto = att_proto.sum(1)
        return att_proto

    def get_featmap(self, input):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        feat_map = self.encoder(input)
        return feat_map

    def pre_encode(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        return x

    def set_module_for_audio(self, args):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
            win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
            n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

        # speechbrain tools 
        self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                            win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                            hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                            n_fft=self.args.extractor.window_size)
        self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)
    
    def set_fea_extractor_for_s2s(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(128) 
    def update_fc_ad(self, dataloader, class_list, session):
        """
        增量会话中更新增量分支参数并使用其原型更新 self.fc 权重。
        使用增量类的支撑集进行微调。
        """
        self.set_session(session)
        if session == 0:
            return
        # 冻结所有其他模块，仅训练增量分支和融合层
        if session % 2 == 1:  # 奇数会话
            # 冻结除增量分支外的所有模块
            for name, module in self.named_children():
                if name not in [ 'gate_fn', 'inc_attn', 'student_model', 'fusion_module']:
                    for p in module.parameters(): 
                        p.requires_grad = False
            
            # 只训练增量相关模块
           
            for p in self.gate_fn.parameters(): 
                p.requires_grad = True
            for p in self.inc_attn.parameters():
                p.requires_grad = True
            for p in self.student_model.parameters(): 
                p.requires_grad = True
            for p in self.fusion_module.parameters():
                p.requires_grad = True

            # 使用增量类的支撑集进行训练
            #optim = torch.optim.SGD([
            #    {'params': self.inc_attn.parameters(), 'lr': self.args.lr.lr_inc2},
            #    {'params': self.gate_fn.parameters(), 'lr': self.args.lr.lr_inc2},
            #    {'params': self.student_model.parameters(), 'lr':1e-4}, 
            #    {'params': self.fusion_module.parameters(), 'lr':1e-4},  
            #], momentum=0.9, weight_decay=1e-4)
            # 1) 只训练增量分支
            optim_inc = torch.optim.SGD([
                {'params': self.inc_attn.parameters(), 'lr': self.args.lr.lr_inc2},
                    {'params': self.gate_fn.parameters(),  'lr': self.args.lr.lr_inc2},
                    ], momentum=0.9, weight_decay=1e-4)

            # 2) 训练学生 & 融合
            optim_fusion = torch.optim.SGD([
                {'params': self.student_model.parameters(), 'lr': 1e-5},
                {'params': self.fusion_module.parameters(), 'lr': 1e-4},
            ] , momentum=0.9, weight_decay=1e-4)

            # 设置为训练模式
          
            self.inc_attn.train()
            self.gate_fn.train()
            self.student_model.train()
            self.fusion_module.train()
            # ============== 关键：一次性重构所有旧类的 teacher & student 特征 ==============
            r_num=5
            t_old, s_old, y_old = self.reconstruct_old_features_dual(session,sample_per_class=r_num)
            for epoch in range(self.args.epochs.epochs_inc):
                for batch in dataloader:
                    data, label = [_.cuda() for _ in batch]
                    data_=data
                    # 确保数据需要梯度
                    data.requires_grad_(True)
                    
                
                    #data= self.encode(data)
                    if self.args.stk:
                        teacher_feat_new = self.encode(data).detach()                              # [B,512]
                        student_feat_new = self.student_model(self.waveform_to_img(data)) # [B,128]

                        data_new,alpha_new = self.fusion_module(teacher_feat_new, student_feat_new,return_alpha=True) # [B,512]
                        if session==1:
                            fused_old, alpha_old = self.fusion_module(t_old[:self.args.num_base*r_num], s_old[:self.args.num_base*r_num],return_alpha=True)
                        else:
                            fused_old1, alpha_old = self.fusion_module(t_old[:self.args.num_base*r_num], s_old[:self.args.num_base*r_num],return_alpha=True)
                            fused_old2 = self.fusion_module(t_old[self.args.num_base*r_num:], s_old[self.args.num_base*r_num:])
                            fused_old= torch.cat([fused_old1, fused_old2], dim=0)


                        min_teacher = getattr(self.args, 'min_teacher', 0.8)
                        lambda_gate = getattr(self.args, 'lambda_gate', 0.2)
                        loss_gate = lambda_gate * ( F.relu(min_teacher - alpha_old).mean()+F.relu(min_teacher - alpha_new).mean() )   #F.relu(min_teacher - alpha_new).mean() 

                        #新类 + 旧类重构融合特征一起，更新 fusion & student & fc ----
                        fused_mix = torch.cat([data_new, fused_old], dim=0)  # 让 fused_new 参与梯度
                        label_mix = torch.cat([label, y_old], dim=0)

                        new_fc = self.update_fc_avg(data_new, label, class_list)
                        old_fc = self.fc.weight[:self.args.num_base + self.args.way * (session//2 ), :].detach()#17session
                        fc = torch.cat([old_fc, new_fc], dim=0)
                        logits = self.get_logits(fused_mix,fc)
                        loss = F.cross_entropy(logits, label_mix)
                        loss=loss_gate+loss
                        # 反向传播更新参数
                        optim_fusion.zero_grad()
                        loss.backward()
                        optim_fusion.step()


                        student_feat_new_inc = self.student_model(self.waveform_to_img(data)).detach()
                        data_new_inc = self.fusion_module(teacher_feat_new, student_feat_new_inc).detach()
                    else:
                        data_new_inc=self.encode(data)
                    # 使用支撑集自身构造支持/查询对
                    support = data_new_inc.view(self.args.episode.episode_shot, self.args.episode.episode_way, -1)  # (shot, way, dim)
                    support = support.permute(1, 0, 2).unsqueeze(0)  # (1, way, shot, dim)
                
                    # 使用支撑集的一部分作为查询集
                    query = support[:, :, :1, :]  # 取每类的第一个样本作为查询
                    query = query.permute(0, 2, 1, 3)  # (1, 1, way, dim)
                    support = support.permute(0,2,1,3)


                     # 前向传播计算损失
                    logits, _, _, _ ,_,_= self._forward(support, query,self.args.stdu.pqa)
                    target = torch.arange(self.args.episode.episode_way).long().cuda().unsqueeze(0)
                    loss_inc = F.cross_entropy(logits, target.view(-1))
                    optim_inc.zero_grad()
                    loss_inc.backward()
                    optim_inc.step()
            
            # 训练完成后设置为评估模式
           
            self.inc_attn.eval()
            self.gate_fn.eval()
            self.student_model.eval()
            self.fusion_module.eval()
            for p in self.gate_fn.parameters(): 
                p.requires_grad = False
            for p in self.inc_attn.parameters():
                p.requires_grad = False
            for p in self.student_model.parameters(): 
                p.requires_grad = False
            for p in self.fusion_module.parameters():
                p.requires_grad = False



            if self.args.stk:            
                student_feat_new = self.student_model(self.waveform_to_img(data)) # [B,128]
                data = self.fusion_module(teacher_feat_new, student_feat_new) # [B,512]
                fused_old = self.fusion_module(t_old, s_old)
                fused_mix = torch.cat([data_new, fused_old], dim=0)
            else:
                data=  data_new_inc
                fused_mix=data_new_inc
                label_mix=label
            # 更新分类器权重
            if not self.args.strategy.data_init:
                new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"), requires_grad=True)
                nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
            else:
                new_fc = self.update_fc_avg(data.detach(), label, class_list)
            
            if 'ft' in self.args.network.new_mode:
                self.update_fc_ft(new_fc, fused_mix.detach(), label_mix, session)


            #if 'ft' in self.args.network.new_mode:
            #    self.update_fc_ft_final(data_, t_old, s_old,label_mix, session,class_list)
              # ----把新类特征统计也更新进去，用于下个 session 再重构时更稳定 ----

            with torch.no_grad():
                self.update_statistics_dual(data_, label)   

        else:  # 偶数会话 - 类减少
            g = self.args.g
            if session == 2:
                self.fc.weight.data[64-g:65-g, :] = 0 #63
                self.fc.weight.data[59-g, :] = 0
                #self.fc.weight.data[58:60, :] = 0

            elif session == 4:
                self.fc.weight.data[69-g:70-g, :] = 0
                self.fc.weight.data[58-g, :] = 0
                #self.fc.weight.data[56:58, :] = 0

            elif session == 6:
                self.fc.weight.data[74-g:75-g, :] = 0 
                self.fc.weight.data[57-g, :] = 0  
                #self.fc.weight.data[54:56, :] = 0

            elif session == 8:
                self.fc.weight.data[79-g:80-g, :] = 0
                self.fc.weight.data[56-g, :] = 0
                # self.fc.weight.data[52:54, :] = 0
            elif session == 10:
                self.fc.weight.data[84-g:85-g, :] = 0
            elif session == 12:
                self.fc.weight.data[88-g:90-g, :] = 0
            elif session == 14:
                self.fc.weight.data[93-g:95-g, :] = 0
            else: 
                self.fc.weight.data[98-g:100-g, :] = 0

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        assert len(data) == self.args.episode.episode_way * self.args.episode.episode_shot
        

        
        if  session % 2 == 1:#奇数
           if not self.args.strategy.data_init:
                 new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"),requires_grad=True)
                 nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
           else:
                 new_fc = self.update_fc_avg(data, label, class_list)

           if 'ft' in self.args.network.new_mode:  # further finetune
                 self.update_fc_ft(new_fc,data,label,session)
        
        else:#类减少微调原型
             #self.update_reduce_ft(data,label,session)
             g=self.args.g
             if session==2:
                self.fc.weight.data[63-g:65-g, :] = 0
             elif session==4:
                self.fc.weight.data[68-g:70-g, :] = 0
             elif session==6:
                self.fc.weight.data[73-g:75-g, :] = 0   
             elif session==8:
                self.fc.weight.data[78-g:80-g, :] = 0
             elif session==10:
                self.fc.weight.data[83-g:85-g, :] = 0
             elif session==12:
                self.fc.weight.data[88-g:90-g, :] = 0
             elif session==14:
                self.fc.weight.data[93-g:95-g, :] = 0
             else: 
                self.fc.weight.data[98-g:100-g, :] = 0
             #self.fc.weight.data[-3:, :] 
             



    
    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            #print(proto)
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc


    def get_logits(self,x,fc):
        
        x = x.float()
        fc = fc.float()
        if 'dot' in self.args.network.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.network.new_mode:
            return self.args.network.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def  update_fc_ft(self,new_fc,data,label,session):
        num_base = self.args.stdu.num_tmpb if self.args.tmp_train else self.args.num_base
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)
        with torch.enable_grad():
            for epoch in range(self.args.epochs.epochs_new):
                #old_fc = self.fc.weight[:num_base + self.args.way * (session - 1), :].detach()
                old_fc = self.fc.weight[:num_base + self.args.way * (session//2 ), :].detach()#17session
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[num_base + self.args.way * (session//2):num_base + self.args.way * (session//2+1), :].copy_(new_fc.data)#17session

    
             
    def update_orgin_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        assert len(data) == self.args.episode.episode_way * self.args.episode.episode_shot
    
        if not self.args.strategy.data_init:
                 new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"),requires_grad=True)
                 nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
                 new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.network.new_mode:  # further finetune
                 self.update_org_fc_ft(new_fc,data,label,session)
        
    


    def update_org_fc_ft(self,new_fc,data,label,session):
        num_base = self.args.stdu.num_tmpb if self.args.tmp_train else self.args.num_base
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs.epochs_new):
                old_fc = self.fc.weight[:num_base + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[num_base + self.args.way * (session - 1):num_base + self.args.way * session, :].copy_(new_fc.data)

    def encode(self, x,return_intermediates=False):
        """
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        """
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        # print(x.shape)
        # print(x[0, 0])          # shape = (63, 128)
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
    # ================== 基础阶段统计（教师 & 学生） ==================
    def compute_base_statistics_dual(self, dataloader, eps=1e-5):
        """
        统计 base session（或你想统计的阶段）里每个类别：
        - 教师特征的均值 / 协方差（512 维）
        - 学生特征的均值 / 协方差（Ds 维）
        """
    
        self.base_mean_t, self.base_cov_t = {}, {}
        self.base_mean_s, self.base_cov_s = {}, {}

        feats_t, feats_s, labels_all = [], [], []
        with torch.no_grad():
            for data, label in dataloader:
                data = data.cuda()
                # 教师特征
                ft = self.encode(data).cpu().numpy()
                feats_t.append(ft)
                # 学生特征
                fs = self.student_model(self.waveform_to_img(data)).cpu().numpy()
                feats_s.append(fs)
                labels_all.append(label.numpy())

        feats_t = np.concatenate(feats_t, 0)
        feats_s = np.concatenate(feats_s, 0)
        labels_all = np.concatenate(labels_all, 0)

        for c in np.unique(labels_all):
            idx = (labels_all == c)
            ct = feats_t[idx]
            cs = feats_s[idx]
            self.base_mean_t[c] = ct.mean(0)
            self.base_cov_t[c]  = np.cov(ct, rowvar=False) + np.eye(ct.shape[1]) * eps

            self.base_mean_s[c] = cs.mean(0)
            self.base_cov_s[c]  = np.cov(cs, rowvar=False) + np.eye(cs.shape[1]) * eps


    # ================== 增量阶段重构（教师 & 学生） ==================
    def reconstruct_old_features_dual(self,session, sample_per_class=5):
        """
        返回：
        t_feat_old: [N_old * K, 512]   （教师重构特征）
        s_feat_old: [N_old * K, Ds]    （学生重构特征）
        labels_old: [N_old * K]
        """
        n_seen_old = self.args.num_base + self.args.way * (session // 2)  # session=1 -> 60
        keys = [c for c in self.base_mean_t.keys() if c < n_seen_old]     # 只拿“旧类”
        t_list, s_list, y_list = [], [], []
        for c in keys:   # 假设和 base_mean_s 的 key 对齐
            mu_t, cov_t = self.base_mean_t[c], self.base_cov_t[c]
            mu_s, cov_s = self.base_mean_s[c], self.base_cov_s[c]
            #alp = getattr(self.args, 'recon_cov_scale', 0.5)  # 0.3~0.7 之间
            #eps = 1e-5
            #cov_t = cov_t * alp + np.eye(cov_t.shape[0]) * eps
            #cov_s = cov_s * alp + np.eye(cov_s.shape[0]) * eps

            # 采样
            t_samples = np.random.multivariate_normal(mu_t, cov_t, sample_per_class)
            s_samples = np.random.multivariate_normal(mu_s, cov_s, sample_per_class)

            t_list.append(t_samples)
            s_list.append(s_samples)
            y_list.extend([c] * sample_per_class)

        t_feat_old = torch.tensor(np.vstack(t_list)).float().cuda()
        s_feat_old = torch.tensor(np.vstack(s_list)).float().cuda()
        labels_old = torch.tensor(y_list).long().cuda()
        return t_feat_old, s_feat_old, labels_old




    # ================== 增量阶段也更新统计 ==================
    def update_statistics_dual(self, new_data, new_labels, lam=0.8, eps=1e-5):
        """
        对当前 session 的新类进行统计，并 EMA 融入已有（如果是旧类也同样 EMA 更新）
        """
        with torch.no_grad():
            # SEE
            ft = self.encode(new_data).cpu().numpy()
            # LPEE
            fs = self.student_model(self.waveform_to_img(new_data)).cpu().numpy()

        new_labels_np = new_labels.cpu().numpy()
        for c in np.unique(new_labels_np):
            idx = (new_labels_np == c)
            ct = ft[idx]
            cs = fs[idx]
            mean_t = ct.mean(0)
            cov_t  = np.cov(ct, rowvar=False) + np.eye(ct.shape[1]) * eps
            mean_s = cs.mean(0)
            cov_s  = np.cov(cs, rowvar=False) + np.eye(cs.shape[1]) * eps

            if hasattr(self, "base_mean_t") and (c in self.base_mean_t):
                self.base_mean_t[c] = lam * self.base_mean_t[c] + (1 - lam) * mean_t
                self.base_cov_t[c]  = lam * self.base_cov_t[c]  + (1 - lam) * cov_t
                self.base_mean_s[c] = lam * self.base_mean_s[c] + (1 - lam) * mean_s
                self.base_cov_s[c]  = lam * self.base_cov_s[c]  + (1 - lam) * cov_s
            else:
                self.base_mean_t[c], self.base_cov_t[c] = mean_t, cov_t
                self.base_mean_s[c], self.base_cov_s[c] = mean_s, cov_s


        
    def waveform_to_img(self, x):
     """
     将 1‑D 波形批量转成 [B,3,H,W] 的 Log‑Mel 图像，
     处理流程与 encode() 的前半段保持一致。
    """
     if x.shape[1] == 44100:
        x = self.fs_spectrogram_extractor(x)
        x = self.fs_logmel_extractor(x)
     elif x.shape[1] == 64000:
        x = self.ns_spectrogram_extractor(x)
        x = self.ns_logmel_extractor(x)
     else:  # 32000
        x = self.ls_spectrogram_extractor(x)
        x = self.ls_logmel_extractor(x)

     x = x.transpose(1, 3)
     x = self.bn0(x)
     x = x.transpose(1, 3)
     x = x.repeat(1, 3, 1, 1)   # 变成 3 通道
     return x                   # shape: [B,3,H,W]

        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn_logit = torch.bmm(q, k.transpose(1, 2))
        attn = attn_logit / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn_logit, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn_logit, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn_logit
    



if __name__ == "__main__":
    proto = torch.randn(25, 512, 2, 4)
    query = torch.randn(75, 512, 2, 4)
