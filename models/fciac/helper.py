# import new Network name here and add in model_class args
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F


def _laplacian_from_proto(proto, sigma=0.5):
    """
    proto: [R, D]  保留类原型
    return: Laplacian L (R×R)
    """
    # 余弦相似度，数值 ∈ [-1,1]
    cos_sim = F.normalize(proto, dim=-1) @ F.normalize(proto, dim=-1).T  # [R,R]
    W = torch.exp(-(1 - cos_sim) / sigma)                                # [R,R]
    D = torch.diag(W.sum(dim=1))
    L = D - W
    return L

def loss_PIP(proto_pre, proto_cur, sigma=0.5):  #PIP loss
    """
    Laplacian distillation loss  ‖L_old - L_new‖_F² / R²
    """
    L_old = _laplacian_from_proto(proto_pre, sigma)
    L_new = _laplacian_from_proto(proto_cur, sigma)
    R = proto_pre.size(0)
    return torch.norm(L_old - L_new, p='fro')**2 / (R * R)

def loss_PIS(proto_forget, proto_retain, m_f=0.5):  #PIS loss
    dmat = torch.cdist(proto_forget, proto_retain, p=2)  # [F,R]
    return F.relu(m_f - dmat).mean()

def get_optimizer_pit(model, args):

    optimizer = torch.optim.SGD([{'params': model.module.encoder.parameters(), 'lr': args.lr.lr_mix_base},
                                    {'params': model.module.transatt_proto.parameters(), 'lr': args.lr.lrg}, 
                                    {'params': model.module.inc_attn.parameters(), 'lr': args.lr.lr_inc}, 
                                    {'params': model.module.gate_fn.parameters(), 'lr': 0.02},
                                    ], # 
                                momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)

    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)

    return optimizer, scheduler

def avg_by_class(q_proto, way):
    """q_proto: [Q*way, 1, D] -> [way, D]"""
    q_proto = q_proto.squeeze(1)          # [Nq, D]
    return q_proto.view(-1, way, q_proto.size(-1)).mean(0)


#PCTS  training strategy
def v_train_pit_final(model, trainloader, optimizer, scheduler, epoch, args):  # 伪增减量训练，双 session，每个 batch 内完成
    tl = Averager()
    ta = Averager()
    treg = Averager()
    model = model.train()
    model.module.set_session(1)
    tqdm_gen = tqdm(trainloader)
    lambda_gate, lambda_T, lambda_F = 0.5, 1.0, 0.8   
    tau = 0.07                            
    m_g,  m_f  = 0.5, 0.5


    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
        samples_per_cls = args.episode2.episode_shot + args.episode2.episode_query
        data = data.view(samples_per_cls, args.episode2.episode_way, -1)
        train_label = train_label.view(samples_per_cls, args.episode2.episode_way)
        audio_samples = data.size(-1)

        # Session 1: 学习五个伪新类
        for param in model.module.encoder.parameters():
            param.requires_grad = True
        base_data = data[:, :args.episode2.base, :].reshape(-1, audio_samples)
        base_feat = model.module.encode(base_data)
        base_lb = train_label[:, :args.episode2.base].reshape(-1)

        syn_new_data = data[:, args.episode2.base:, :].view(samples_per_cls, 2, args.episode2.syn_new, -1)
        lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
        mixed_data = lam * syn_new_data[:, 0, :, :] + (1 - lam) * syn_new_data[:, 1, :, :]
        mixed_data = mixed_data.reshape(-1, audio_samples)
        mixed_feat = model.module.encode(mixed_data)

        syn_proto = mixed_feat.view(samples_per_cls, args.episode2.syn_new, -1)[:args.episode2.episode_shot, :, :].mean(0)
        picked_new_cls = torch.Tensor(np.random.choice(args.num_all - args.num_base, 5, replace=False) + args.num_base).long()
        novel_mask = torch.zeros((args.num_all - args.num_base, model.module.num_features))
        novel_mask[picked_new_cls - args.num_base, :] = syn_proto.cpu()
        model.module.fc.weight.data[args.num_base:, :] = novel_mask.cuda()

        base_feat = base_feat.unsqueeze(0).unsqueeze(0)
        proto = model.module.fc.weight[:, :].detach().unsqueeze(0).unsqueeze(0)
        mix_query = mixed_feat[args.episode2.episode_shot * args.episode2.syn_new:, :].unsqueeze(0).unsqueeze(0)
        query_feat = torch.cat([base_feat, mix_query], dim=2)
        query_logits, anchor_loss, _, proto_1,slf_1, inc_1= model.module._forward(proto, query_feat, args.stdu.pqa, None, None)
        proto_1 = avg_by_class(proto_1, args.num_all)
        syn_lbs = torch.tile(picked_new_cls, (args.episode2.episode_query,)).cuda()
        logits_1 =  query_logits
        labels_1 = torch.cat([base_lb, syn_lbs], dim=0)
        loss1_cls  = F.cross_entropy(logits_1, labels_1)    ### <<< NEW <<<
        loss_s1 = loss1_cls #+ lambda_gate * loss1_gate               ### <<< NEW <<<
        #loss1 = F.cross_entropy(logits_1, labels_1) + anchor_loss + mix_anchor_loss
        optimizer.zero_grad()
        loss_s1.backward()
        optimizer.step()
        forget_classes = picked_new_cls[-2:]
        retain_classes = list(range(args.num_base)) + picked_new_cls[:3].tolist()
        proto_retain_pre =  proto_1[retain_classes].detach().clone()
        proto_forget = proto_1[forget_classes].detach().clone()

        # Session 2: 遗忘两个伪新类，仅保留3个伪类和所有基类
        model.module.set_session(2)
        for param in model.module.encoder.parameters():
            param.requires_grad = False
    
        for cls in forget_classes:
            model.module.fc.weight.data[cls] = 0
        
        # 放到 Session 2 中（在清零 fc 后）
        proto = model.module.fc.weight[:, :].detach().unsqueeze(0).unsqueeze(0)

        # 构造 retain 类的混合数据（重新 mixup）
        syn_new_data = data[:, args.episode2.base:args.episode2.base + 6, :].view(samples_per_cls, 2, 3, -1)
        lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
        mixed_data = lam * syn_new_data[:, 0, :, :] + (1 - lam) * syn_new_data[:, 1, :, :]
        mixed_data = mixed_data.reshape(-1, audio_samples)
        mixed_feat = model.module.encode(mixed_data)

        # 重新 encode base
        base_feat = model.module.encode(base_data).unsqueeze(0).unsqueeze(0)
        #base_logits, _, _, proto_b2 , slf_b2, inc_b2= model.module._forward(proto, base_feat, args.stdu.pqa, None, None)

        # mix query
        mix_query = mixed_feat[args.episode2.episode_shot * 3:, :].unsqueeze(0).unsqueeze(0)
        #mix_logits, _, _, proto_m2 , slf_m2, inc_m2= model.module._forward(proto, mix_query, args.stdu.pqa, None, None)

        query_feat = torch.cat([base_feat, mix_query], dim=2)
        query_logits, anchor_loss, _, proto_2,slf_2, inc_2= model.module._forward(proto, query_feat, args.stdu.pqa, None, None)
        proto_2 = avg_by_class(proto_2, args.num_all)
        
        # 构建标签
        syn_lbs = torch.tile(picked_new_cls[:3], (args.episode2.episode_query,)).cuda()
        labels_2= torch.cat([base_lb, syn_lbs], dim=0)
        logits_2 = query_logits
        loss2_cls  = F.cross_entropy(logits_2, labels_2)       
        proto_retain_cur=proto_2[retain_classes]
        loss_pip  = loss_PIP(proto_retain_pre, proto_retain_cur)               
        loss_pis  = loss_PIS(proto_forget,proto_retain_cur,  m_f=m_f)          
        loss_s2 = (loss2_cls  +lambda_F   * loss_pis +lambda_T   * loss_pip

                   )  
        optimizer.zero_grad()
        loss_s2.backward()
        optimizer.step()
        total_loss = loss_s1.item() + loss_s2.item()
        acc = count_acc(logits_2[:, retain_classes],
                torch.tensor([retain_classes.index(l.item()) for l in labels_2]).cuda())
        tl.add(total_loss)
        ta.add(acc)
    return tl.item(), ta.item(), 0

def replace_base_fc(trainset, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding= model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    for class_index in range(args.num_base):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)
    model.module.fc.weight.data[:args.num_base] = proto_list
    return model

def mixup_feat(feat, gt_labels, alpha=1.0):
    if alpha > 0:
        lam = alpha
    else:
        lam = 0.5
    batch_size = feat.size()[0]
    index = torch.randperm(batch_size).to(device=feat.device)
    mixed_feat = lam * feat + (1 - lam) * feat[index, :]
    gt_a, gt_b = gt_labels, gt_labels[index]
    return mixed_feat, gt_a, gt_b, lam