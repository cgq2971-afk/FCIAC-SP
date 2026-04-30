import os
import time
from copy import deepcopy
from .helper import get_optimizer_pit,replace_base_fc,v_train_pit_final
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from dataloader.dataloader import get_dataloader, get_dataset_for_data_init, get_pretrain_dataloader, get_testloader,get_incre_dataloader,get_base_dataloader_meta,custom_collate
#from models.base.fscil_trainer import FSCILTrainer as Trainer
from models.fciac.incremental_train_helper import base_train, get_optimizer_incremental
from models.fciac.standard_train_helper import get_optimizer_standard, standard_base_train, standard_test
from utils.utils import AverageMeter, Averager, DAverageMeter, acc_utils, count_acc, count_per_cls_acc, save_list_to_txt,cd_acc_utils,get_torch_size
from .FCIAC_Net import MYNET
from models.base.base import Trainer

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        print("*********fciac init***************")
        self.args = args
        self.set_up_model()
    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.network.base_mode)
        #print(MYNET)
        #print(self.model)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        
        if self.args.model_dir.stdu_model_dir != None:
            print('Loading init parameters from: %s' % self.args.model_dir.stdu_model_dir)
            self.best_model_dict = torch.load(self.args.model_dir.stdu_model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')

            pass

    
    def train(self):
        if self.args.model_dir.stdu_model_dir != None:
            self.set_up_datasets(self.args)
            self.best_model_dict = torch.load(self.args.model_dir.stdu_model_dir)['params']
        else:
            self.fciac_train()

        # 基础训练完成后新增调用：存储旧类统计量
        # 例：standard_train 或 pit_train 之后
        train_set, _, base_loader, _ = get_base_dataloader_meta(self.args)
        self.model.module.compute_base_statistics_dual(base_loader)


        for session in range(self.args.start_session, self.args.num_session):
            acc_dict_aver = DAverageMeter()
            tsl_aver = AverageMeter()
            tsa_aver = AverageMeter()
            data_dict = {}
            data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
                = get_dataloader(self.args, session)
            data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

            self.model = self.update_param(self.model, self.best_model_dict)
            #self.model.module.set_session(session)

            if session == 0:  # load base class train img label           
                # always replace fc with avg mean and save the replaced model
                self.data_init(data_dict, session)

                self.model.module.mode = 'avg_cos'
                tsl, tsa, acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], session, data_dict['trainloader'])

                self.sess_acc_dict[f'sess {session}'] = acc_dict
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                # print(acc_dict)
                print(cls_sample_count)
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("Inference session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)
                self.model.module.mode = self.args.network.new_mode
                self.model.eval()
                #if session % 2 == 0:
                      #get_dataloader(self.args, session)
                      #self.model.module.update_fc(data_dict['trainloader'], np.unique(data_dict['train_set'].targets), session)
                times = self.args.test_times if session % 2 == 1 else 1
                for i in range(times):
                    get_dataloader(self.args, session)
                    #if session % 2 == 1:
                    self.model.module.update_fc_ad(data_dict['trainloader'],np.unique(data_dict['train_set'].targets), session)
                    tmp_tsl, tmp_tsa, tmp_acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], session, data_dict['trainloader'])
                    tsl_aver.update(tmp_tsl)
                    tsa_aver.update(tmp_tsa)
                    acc_dict_aver.update(tmp_acc_dict)
                tsl = tsl_aver.average()
                tsa = tsa_aver.average()
                acc_dict = acc_dict_aver.average()

                self.sess_acc_dict[f'sess {session}'] = acc_dict

                # save model
                self.save_model(tsa, session)
                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        print(cls_sample_count)
        t_end_time = time.time()
        output = self.pretty_output()

        self.result_list.append(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}, \
            Base Session Best acc:{self.trlog['max_acc']}")
        self.result_list.append(self.sess_acc_dict)
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)

        print(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}, \
            Base Session Best acc:{self.trlog['max_acc']}")  
             
# 蒸馏训练（在fscil_trainer.py新增函数，pit_train之后调用）
    def distill_student(self, teacher_model, student_model, dataloader, epochs=50):
        """
        teacher_model: 一般传 self.model.module.encoder（但我们不直接调用它，而是走 self.model.module.encode）
        student_model: 你的轻量学生（输出 128 维）
        dataloader: 建议用 base 的大批量数据（非 episodic），比如 standard/base train 的 trainloader
        """
    
    

        # 模式：teacher eval（冻结）；学生 & 投影 train
        teacher_model.eval()
        student_model.train().cuda()
     # 拿到融合里的 projection（把 128 → 512），加入优化器
        student_proj = self.model.module.fusion_module.student_proj  # Linear/MLP → 512
        student_proj.train()

        # 优化器：学生 + 投影
        optimizer = torch.optim.Adam(
            [
                {"params": student_model.parameters(), "lr": 5e-4},
                {"params": student_proj.parameters(),  "lr": 5e-4},
            ],
            weight_decay=1e-4
        )

        scaler = GradScaler()  # 可选的混合精度

        for epoch in range(epochs):
            for data, _ in dataloader:
                data = data.cuda(non_blocking=True)

                # 保证 encode 的输入形状正确：期望 [B, T]
                if data.dim() == 3 and data.size(1) == 1:
                    data_wave = data.squeeze(1)     # [B, T]
                else:
                    data_wave = data                 # [B, T] 就保持不变

                with torch.no_grad():
                    # 教师特征：走 encode，包含前端 & bn0，对齐 base 训练分布
                    feat_teacher = self.model.module.encode(data_wave)   # [B, 512] 已经池化

                # 学生输入：走已有的 waveform_to_img（或学生自己的前端）
                data_img = self.model.module.waveform_to_img(data)       # 若 data 已是 [B,T]，内部应处理好维度
                feat_student = student_model(data_img)                    # [B, 128]
                feat_student_512 = student_proj(feat_student)             # [B, 512]

                # 归一化 + 混合蒸馏损失：cosine + MSE（可调权重）
                t = F.normalize(feat_teacher, p=2, dim=1)
                s = F.normalize(feat_student_512, p=2, dim=1)
                loss_cos = 1.0 - (t * s).sum(dim=1).mean()
                loss_mse = F.mse_loss(feat_student_512, feat_teacher)
                loss = 0.5 * loss_mse + 0.5 * loss_cos

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fciac_train(self):
        self.args.tmp_train = True
        if self.args.model_dir.tmp_model_dir is not None:
            self.load_model(self.args.model_dir.tmp_model_dir)
            self.standard_train(temp=True, pretrained=True)
        else:
            self.standard_train(temp=True)

        self.reset_trlog(self.args.stdu.num_tmps)
        self.incremental_train()

        self.args.tmp_train = False
        self.reset_trlog(self.args.num_session)
        if self.args.model_dir.s0_model_dir is not None:
            self.load_model(self.args.model_dir.s0_model_dir)
            self.standard_train(temp=False, pretrained=True)
        else:
            self.standard_train(temp=False)
        self.pit_train()
        


    def pit_train(self):
        session=0
        optimizer, scheduler = get_optimizer_pit(self.model, self.args)
        train_set, valset, trainloader, testloader=get_base_dataloader_meta(self.args)
        self.model = self.update_param(self.model, self.best_model_dict)
        self.model.module.inc_attn.load_state_dict(self.model.module.slf_attn.state_dict())
        for param in self.model.module.slf_attn.parameters():
            param.requires_grad = False
        print('new mix_up_classes for this session:\n', np.unique(train_set.targets))
        for epoch in range(self.args.epochs.epochs_base):
            start_time = time.time()
            tl, ta, treg = v_train_pit_final(self.model, trainloader, optimizer, scheduler, epoch, self.args)
            # test model with all seen class
            tsl, tsa, acc_dict = self.incre2_test(self.model, testloader,session)
            self.sess_acc_dict[f'sess {session}'] = acc_dict

            # save better model
            if (tsa * 100) >= self.trlog['max_acc'][session]:
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['max_acc_epoch'] = epoch
                save_model_dir = os.path.join(self.args.save_path, 'mix_session' + str(session) + '_max_acc.pth')
                # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                # torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('********A better model is found!!**********')
                print('Saving model to :%s' % save_model_dir)
            print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                        self.trlog['max_acc'][session]))

            self.trlog['train_loss'].append(tl)
            self.trlog['train_acc'].append(ta)
            self.trlog['test_loss'].append(tsl)
            self.trlog['test_acc'].append(tsa)
            lrc = scheduler.get_last_lr()[0]
            self.result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
            print('epoch:%03d,lr:%.4f,training_ce_loss:%.5f, training_reg_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, treg,ta, tsl, tsa))
            print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\nstill need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (self.args.epochs.epochs_base - epoch) / 60))
            scheduler.step()
                    
            self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            ##蒸馏学生模型
           
        base_loader = get_base_dataloader_meta(self.args)[2]  # trainloader
        self.distill_student(self.model.module.encoder, self.model.module.student_model, base_loader)    

        if self.args.strategy.data_init:
            train_set, valset,trainloader, valloader = get_base_dataloader_meta(self.args)
            #print("Went inside #################")
            print("Updating old class with class means ")
            self.model.load_state_dict(self.best_model_dict)
            self.model = replace_base_fc(train_set, self.model, self.args)
            best_model_dir = os.path.join(self.args.save_path, 'mixsession0'+ '_max_acc.pth')
            print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            session0_best_model_dict = deepcopy(self.model.state_dict())
            # torch.save(dict(params=self.model.state_dict()), best_model_dir)

            self.model.module.mode = 'avg_cos'
            tsl, tsa, acc_dict = self.incre2_test(self.model, valloader, 0)

            self.sess_acc_dict[f'sess 0'] = acc_dict
            if (tsa * 100) >= self.trlog['max_acc'][0]:
                self.trlog['max_acc'][0] = float('%.3f' % (tsa * 100))
                print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][0]))
    
    


    def standard_train(self, temp=False, pretrained=False):
        session = 0
        data_dict = {}
        data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
            = get_pretrain_dataloader(self.args)
        data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

        net_dict = {}
        if temp:
            print('==> Classes for this TEMPORARY standard train stage:\n', np.unique(data_dict['train_set'].targets))
        else:
            print('==> Classes for this standard train stage:\n', np.unique(data_dict['train_set'].targets))

        net_dict['optimizer'], net_dict['scheduler'] = get_optimizer_standard(self.model, self.args)

        """****************train and val*************************"""
        if not pretrained:
            for epoch in range(self.args.epochs.epochs_std):
                std_start_time = time.time()
                tl, ta = standard_base_train(self.args, self.model, data_dict['trainloader'], net_dict['optimizer'], net_dict['scheduler'], epoch, temp)
                net_dict['epoch'] = epoch
                res_dict = {'tl': tl, 'ta': ta}
                tsl, tsa, acc_dict, cls_sample_count = standard_test(self.args, self.model, data_dict['testloader'], epoch, session, temp)
                # set save path
                if temp:
                    save_model_path = os.path.join(self.args.save_path, f'temp_std_train{self.args.epochs.epochs_std}_max_acc.pth')
                else:
                    save_model_path = os.path.join(self.args.save_path, f'std_train{self.args.epochs.epochs_std}_max_acc.pth')
                self.save_better_model(tsa, net_dict, session, save_model_path)
                self.record_info(tsa, tsl, net_dict, res_dict, std_start_time, self.args.epochs.epochs_std)
                net_dict['scheduler'].step()

            """****************record on best model*************************"""
            stage_flag = "TEMPORARY(not load)" if temp else "Standard(not load)"
            self.result_list.append('{} standard train stage, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        stage_flag, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            print('{} test loss={:.3f}, test acc={:.3f}'.format(stage_flag, tsl, tsa))

        """****************data init and test again*************************"""
        if self.args.strategy.data_init:
            #data init and replace the model
            self.data_init(data_dict, session)
            self.model.module.mode = 'avg_cos'
            tsl, tsa, acc_dict, cls_sample_count = standard_test(self.args, self.model,data_dict['testloader'], 0, session, temp)
            print(cls_sample_count)
            # self.sess_acc_dict[f'sess {session}'] = acc_dict
            if (tsa * 100) >= self.trlog['max_acc'][session]:
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                if temp:
                    print('The NEW(after data init) best test acc of TEMPORARY standard train stage={:.3f}'.format(self.trlog['max_acc'][session]))
                else:
                    print('The NEW(after data init) best test acc of standard train stage={:.3f}'.format(self.trlog['max_acc'][session]))

        if temp:
            self.result_list.append(f"==> TEMPORARY standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
                Best acc:{self.trlog['max_acc']}")
        else:
            self.result_list.append(f"==> Standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
                Best acc:{self.trlog['max_acc']}")
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)

    def incremental_train(self):
            session = 0
            data_dict = {}
            data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
                 = get_dataloader(self.args,session)
            
            data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

            self.model = self.update_param(self.model, self.best_model_dict)

            print('new classes for this session:\n', np.unique(data_dict['train_set'].targets))
            optimizer, scheduler = get_optimizer_incremental(self.model, self.args)

            for epoch in range(self.args.epochs.epochs_stdu_base):
                start_time = time.time()
                # train base sess
                self.model.eval()
                tl, ta = base_train(self.model, data_dict['trainloader'], optimizer, scheduler, epoch, self.args)
                self.model = self.replace_base_fc(get_dataset_for_data_init(self.args), self.model)
                self.model.module.mode = 'avg_cos'
                # prepare to validate
                net_dict = {'optimizer': optimizer, 'scheduler': scheduler, 'epoch': epoch}
                res_dict = {'result_list': self.result_list, 'tl': tl, 'ta': ta}
                vl, va, acc_dict, cls_sample_count = self.validate(session)
                save_model_dir = os.path.join(self.args.save_path, f'incremental_train{self.args.epochs.epochs_stdu_base}_max_acc.pth')
                self.save_better_model(va, net_dict, session, save_model_dir)
                self.record_info(va, vl, net_dict, res_dict, start_time, self.args.epochs.epochs_stdu_base)  
                net_dict['scheduler'].step()               
            # always replace fc with avg mean and save the replaced model
            self.data_init(data_dict, session)

            self.model.module.mode = 'avg_cos'
            tsl, tsa, acc_dict, cls_sample_count = self.incre_test(self.model, data_dict['testloader'], session, data_dict['trainloader'])
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.result_list.append('Incremental train, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

            print(acc_dict)
            print(cls_sample_count)
            print('The test acc of incremental train={:.3f}'.format(self.trlog['max_acc'][session]))

    def reset_trlog(self, sessions):
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * sessions

    def validate(self, session):
        print('>>>= Validation stage of incremental train')
        with torch.no_grad():
            model = self.model
            for session in range(1, 9):
                 
                train_set, valset, trainloader, valloader = get_incre_dataloader(self.args, session)
                # trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_orgin_fc(trainloader, np.unique(train_set.targets), session)
                vl, va, acc_dict, cls_sample_count = self.incre_test(model, valloader,  session, trainloader)
                print('Validation of incremental train: Session {}, total loss={:.4f}, acc={:.4f}'.format(session, vl, va))
        return vl, va, acc_dict, cls_sample_count

    def test(self, model, testloader,  session, trainloader):

        
        if session % 2 == 1:  # session 为奇数时
            test_class = self.args.num_base + (session//2+1) * self.args.way
        else:  # session 为偶数时
            if session==0:
               test_class = self.args.num_base + ((session//2) * self.args.way)
            else:
               test_class = self.args.num_base + ((session//2) * self.args.way)-1
        model = model.eval()

        vl = Averager()
        va = Averager()
        da = DAverageMeter()
        ca = DAverageMeter()

        sup_emb, novel_ids = None, None
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                if session >0 and self.args.stk:
                     model.module.mode = 'fusion_encoder'
                else:
                     model.module.mode = 'encoder'
                #model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                proto = model.module.fc.weight[:test_class, :].detach().unsqueeze(0).unsqueeze(0)
                logits, anchor_loss, pqa_query, pqa_proto,_,_ = model.module._forward(proto, query,self.args.stdu.pqa, sup_emb, novel_ids)
    
                #print("test_label min:", test_label.min().item())
                #print("test_label max:", test_label.max().item())

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                #unique_labels = torch.unique(test_label)
                #print(unique_labels)

                per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
                    
                vl.add(loss.item())
                va.add(acc)
                da.update(per_cls_acc)
                ca.update(cls_sample_count)
            vl = vl.item()
            va = va.item()
            da = da.average()
            #print(da)
            ca = ca.average()
            acc_dict = cd_acc_utils(da, self.args.num_base, self.args.num_session, self.args.way, session,self.args.g)
        print(acc_dict)
        return vl, va, acc_dict, ca
    
    def incre_test(self, model, testloader,  session, trainloader):
        test_class = self.args.num_base + session * self.args.way
        model = model.eval()

        vl = Averager()
        va = Averager()
        da = DAverageMeter()
        ca = DAverageMeter()

        sup_emb, novel_ids = None, None
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                proto = model.module.fc.weight[:test_class, :].detach().unsqueeze(0).unsqueeze(0)
                self.model.module.set_session(0)
                logits, anchor_loss, pqa_query, pqa_proto,_,_ = model.module._forward(proto, query, self.args.stdu.pqa, sup_emb, novel_ids)
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                da.update(per_cls_acc)
                ca.update(cls_sample_count)
            vl = vl.item()
            va = va.item()
            da = da.average()
            ca = ca.average()
            acc_dict = acc_utils(da, self.args.num_base, self.args.num_session, self.args.way, session)
        print(acc_dict)
        return vl, va, acc_dict, ca
    
    def incre2_test(self, model, testloader,session):
        test_class = self.args.num_base + session * self.args.way
        model = model.eval()

        vl = Averager()
        va = Averager()
        da = DAverageMeter()
        ca = DAverageMeter()

        sup_emb, novel_ids = None, None
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                proto = model.module.fc.weight[:test_class, :].detach().unsqueeze(0).unsqueeze(0)
                self.model.module.set_session(1)
                logits, anchor_loss, pqa_query, pqa_proto,_,_ = model.module._forward(proto, query,self.args.stdu.pqa, sup_emb, novel_ids)
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                da.update(per_cls_acc)
                ca.update(cls_sample_count)
            vl = vl.item()
            va = va.item()
            da = da.average()
            ca = ca.average()
            acc_dict = acc_utils(da, self.args.num_base, self.args.num_session, self.args.way, session)
        print(acc_dict)
        return vl, va, acc_dict
    