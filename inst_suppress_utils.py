from utils import train_diff_transform, ToTensor_transform
import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import time

def get_batch_idx_group(total_num, batch_size=512, shuffle=False, drop_last=False):
    if shuffle:
        perm_id = np.random.permutation(total_num)
    else:
        perm_id = np.arange(total_num)
    if drop_last:
        batch_num = total_num // batch_size
    else:
        batch_num = (total_num - 1) // batch_size + 1
    batch_idx_list = []
    for i in range(batch_num):
        batch_idx_list.append(perm_id[i*batch_size:min(total_num, (i+1)*batch_size)])
    return batch_idx_list

def get_total_batch_num(total_num, batch_size=512, drop_last=False):
    if drop_last:
        batch_num = total_num // batch_size
    else:
        batch_num = (total_num - 1) // batch_size + 1
    return batch_num

def get_batch_DBindex(net, pos_1, target, use_out=False, DBindex_use_org_sample=False, curriculum=''):
    net.eval()
    org_sample = pos_1.cuda(non_blocking=True)
    aug_bank = []
    aug_out_bank = []
    label_bank = []
    GT_target_bank = []
    if curriculum != 'DBindex_cluster_GT_org_sample_only':
        for i in range(5):
            aug_sample = train_diff_transform(org_sample)
            feature, out = net(aug_sample)

            if use_out:
                aug_out_bank.append(out)
            else:
                aug_bank.append(feature)

            label_bank.append(np.arange(org_sample.shape[0]))
            GT_target_bank.append(target.cpu().detach().numpy())

            # print(curriculum)
            # input('check1')

    if DBindex_use_org_sample or curriculum == 'DBindex_cluster_GT_org_sample_only':
        # input('check2')
        feature, out = net(org_sample)
        if use_out:
            aug_out_bank.append(out)
        else:
            aug_bank.append(feature)
        label_bank.append(np.arange(org_sample.shape[0]))
        GT_target_bank.append(target.cpu().detach().numpy())
    
    if use_out:
        aug_bank = torch.cat(aug_out_bank, dim=0).cpu().detach().numpy()
    else:
        aug_bank = torch.cat(aug_bank, dim=0).cpu().detach().numpy()
    # aug_out_bank = torch.cat(aug_out_bank, dim=0).cpu().detach().numpy()
    label_bank = np.concatenate(label_bank, axis=0)
    GT_target_bank = np.concatenate(GT_target_bank, axis=0)

    # print(metrics.davies_bouldin_score(aug_bank, label_bank))
    # print(metrics.davies_bouldin_score(aug_out_bank, label_bank))
    if curriculum == 'DBindex_cluster_GT_org_sample_only':
        return 0.0, metrics.davies_bouldin_score(aug_bank, GT_target_bank)
    else:
        return metrics.davies_bouldin_score(aug_bank, label_bank), metrics.davies_bouldin_score(aug_bank, GT_target_bank)

def get_inst_sigma(net, pos_1, target, use_out=False):
    net.eval()
    org_sample = pos_1.cuda(non_blocking=True)
    aug_bank = []
    aug_out_bank = []
    for i in range(5):
        aug_sample = train_diff_transform(org_sample)
        feature, out = net(aug_sample)

        if use_out:
            aug_out_bank.append(out)
        else:
            aug_bank.append(feature)
    
    if use_out:
        aug_bank = torch.stack(aug_out_bank, dim=2)
    else:
        aug_bank = torch.stack(aug_bank, dim=2)
    # aug_out_bank = torch.cat(aug_out_bank, dim=0).cpu().detach().numpy()

    aug_center = torch.mean(aug_bank, dim=2).unsqueeze(2).repeat((1,1,5))

    # temp1 = (aug_bank - aug_center) ** 2
    # temp2 = torch.sum((aug_bank - aug_center) ** 2, dim=1)
    # temp3 = torch.sqrt(torch.sum((aug_bank - aug_center) ** 2, dim=1))
    inst_sigma = torch.mean(torch.sqrt(torch.sum((aug_bank - aug_center) ** 2, dim=1)), dim=1)

    # input(inst_sigma.shape)

# for i in range(c):
#     idx_i = np.where(train_targets == i)[0]
#     class_i = train_data[idx_i,:,0,0]
#     class_i_center = train_data[idx_i].mean(axis=0)[:,0,0]
#     class_center.append(class_i_center)
#     sort_class.append(class_i)
#     intra_class_dis.append(np.mean(np.sqrt(np.sum((class_i-class_i_center)**2, axis=1))))

    return inst_sigma.cpu().detach().numpy()

def get_decrease_inst_sigma_id(net, batch_idx_list, data_loader, reverse, use_out=False):
    batch_idx_bar = tqdm(batch_idx_list, desc="Reordering")
    inst_sigma = []
    for batch_idx in batch_idx_bar:
        pos_1, target = data_loader.get_batch(batch_idx)
        inst_sigma.append(get_inst_sigma(net, pos_1, target, use_out))
    
    inst_sigma = np.concatenate(inst_sigma, axis=0)
    if reverse:
        inst_sigma_decrease_idx = np.argsort(inst_sigma)
    else:
        inst_sigma_decrease_idx = np.argsort(-inst_sigma)

    return inst_sigma_decrease_idx

def get_scheduler(new_batch_idx_list, epoch, whole_epoch, batch_num, start_batch_num_ratio=0, scheduler_curve='0_0.5_1', flag_shuffle_new_batch_list=False):

    # batch_num = len(new_batch_idx_list)
    if scheduler_curve == '0_0.5_1':
        schedule_batch_num = batch_num * (epoch - 1) // whole_epoch + 1
        schedule_batch_num = int((1 - start_batch_num_ratio) * schedule_batch_num + start_batch_num_ratio * batch_num)
    elif scheduler_curve == '0_1_1':
        schedule_batch_num = batch_num * (epoch - 1) // (whole_epoch * 0.5) + 1
        schedule_batch_num = int((1 - start_batch_num_ratio) * schedule_batch_num + start_batch_num_ratio * batch_num)

    new_batch_idx_list = new_batch_idx_list[:schedule_batch_num]
    if not flag_shuffle_new_batch_list: 
        return new_batch_idx_list
    else:
        # shuffle batch order instead of the order that we use.
        shuffle_new_batch = np.random.permutation(len(new_batch_idx_list))

        shuffle_new_batch_list = []

        for i in range(len(shuffle_new_batch)):
            idx = shuffle_new_batch[i]
            shuffle_new_batch_list.append(new_batch_idx_list[idx])

        return shuffle_new_batch_list

def get_scheduler_length(batch_num, epoch, whole_epoch, start_batch_num_ratio=0, scheduler_curve='0_0.5_1'):

    if scheduler_curve == '0_0.5_1':
        schedule_batch_num = batch_num * (epoch - 1) // whole_epoch + 1
        schedule_batch_num = int((1 - start_batch_num_ratio) * schedule_batch_num + start_batch_num_ratio * batch_num)
    elif scheduler_curve == '0_1_1':
        schedule_batch_num = batch_num * (epoch - 1) // (whole_epoch * 0.5) + 1
        schedule_batch_num = int((1 - start_batch_num_ratio) * schedule_batch_num + start_batch_num_ratio * batch_num)

    return schedule_batch_num

def compare(a1, a2):
    reverse_num = 0
    for i in range(len(a1)):
        for j in range(i+1, len(a1)):
            figure1 = a1[i]
            figure2 = a1[j]
            for k in range(len(a2)):
                if a2[k] == figure1:
                    break
                elif a2[k] == figure2:
                    reverse_num += 1
                    break
    return reverse_num

def sample_from_mass(net, data_loader, epoch, batch_size, scheduler_length, candidate_pool_size, use_out=False, reverse=False, curriculum='', mass_candidate='mass_candidate', all_in=False, random_last_3batch=False, last_one_copy_previous=False):
    if all_in:
        # print('here')
        # input()
        return get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=batch_size, shuffle=True, drop_last=True)
        
    arange_batch_idx = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=batch_size, shuffle=False, drop_last=True)
    batch_num = len(arange_batch_idx)
    feature_bank = []
    label_bank = []
    print("Curriculum sample_from_mass: extracting feature..")
    net.eval()
    for i in range(batch_num):
        batch_feature = []
        pos_1, target = data_loader.get_batch(arange_batch_idx[i])
        pos_1 = pos_1.cuda(non_blocking=True)
        for _ in range(5):
            pos_1_aug = train_diff_transform(pos_1)
            feature, out = net(pos_1_aug)
            if not use_out:
                batch_feature.append(feature.cpu().detach().numpy())
            else:
                batch_feature.append(out.cpu().detach().numpy())
        batch_feature = np.stack(batch_feature, axis=1)
        feature_bank.append(batch_feature)
        label_bank.append(target.cpu().detach().numpy())
    feature_bank = np.concatenate(feature_bank, axis=0)
    label_bank = np.concatenate(label_bank, axis=0)

    print("Curriculum sample_from_mass: extracting feature DONE.")

    print("Curriculum sample_from_mass: resampling..")

    time_test = 0.0

    tobe_batched_idx = np.arange(data_loader.data_source.data.shape[0])
    new_sampled_batch = []
    new_sampled_DBindex = []
    for i in range(scheduler_length):
        if random_last_3batch and i >= batch_num - 3:
            batched_id_candidate_sub = get_batch_idx_group(len(tobe_batched_idx), batch_size=batch_size, shuffle=True, drop_last=True)
            for batch_id_sub in batched_id_candidate_sub:
                batch_id = tobe_batched_idx[batch_id_sub]

                # batched_feature = feature_bank[batch_id, :, :].reshape((-1, feature_bank.shape[2]))

                # if curriculum in ['DBindex_high2low', 'DBindex_ratio_inst_cluster_GT', 'DBindex_product_inst_cluster_GT']:
                #     batched_label_inst = np.tile(np.arange(batch_size), (5,1)).transpose().reshape((-1))
                #     inst_DB = metrics.davies_bouldin_score(batched_feature, batched_label_inst)
                # if curriculum in ['DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 'DBindex_product_inst_cluster_GT', 'DBindex_cluster_GT_org_sample_only']:
                #     batched_label = np.tile(label_bank[batch_id], (5,1)).transpose().reshape((-1))
                #     cluster_GT = metrics.davies_bouldin_score(batched_feature, batched_label)

                # if curriculum == "DBindex_high2low":
                #     new_sampled_DBindex.append(inst_DB)
                # elif curriculum == "DBindex_cluster_GT":
                #     new_sampled_DBindex.append(-cluster_GT)
                # elif curriculum == "DBindex_ratio_inst_cluster_GT":
                #     new_sampled_DBindex.append(inst_DB / cluster_GT)
                # elif curriculum == "DBindex_product_inst_cluster_GT":
                #     new_sampled_DBindex.append(inst_DB * cluster_GT)
                # elif curriculum == "DBindex_cluster_GT_org_sample_only":
                #     new_sampled_DBindex.append(-cluster_GT)

                new_sampled_batch.append(batch_id)
            break

        elif i == batch_num - 1 and i >= 1 and last_one_copy_previous:
            new_sampled_batch.append(new_sampled_batch[0])
            new_sampled_DBindex.append(new_sampled_DBindex[0])

        else:
            candidate_batch_list = []
            DBindex_list =[]
            candidate_batch_sub = []
            for _ in range(candidate_pool_size):
                batched_id_candidate_sub = get_batch_idx_group(len(tobe_batched_idx), batch_size=batch_size, shuffle=True, drop_last=True)
                for j in range(len(batched_id_candidate_sub)):
                    batched_id_candidate = tobe_batched_idx[batched_id_candidate_sub[j]]
                    batched_feature = feature_bank[batched_id_candidate, :, :].reshape((-1, feature_bank.shape[2]))

                    if curriculum in ['DBindex_high2low', 'DBindex_ratio_inst_cluster_GT', 'DBindex_product_inst_cluster_GT']:
                        batched_label_inst = np.tile(np.arange(batch_size), (5,1)).transpose().reshape((-1))
                        inst_DB = metrics.davies_bouldin_score(batched_feature, batched_label_inst)
                    if curriculum in ['DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 'DBindex_product_inst_cluster_GT', 'DBindex_cluster_GT_org_sample_only']:
                        batched_label = np.tile(label_bank[batched_id_candidate], (5,1)).transpose().reshape((-1))
                        cluster_GT = metrics.davies_bouldin_score(batched_feature, batched_label)

                    candidate_batch_list.append(batched_id_candidate)
                    candidate_batch_sub.append(batched_id_candidate_sub[j])

                    if curriculum == "DBindex_high2low":
                        DBindex_list.append(inst_DB)
                    elif curriculum == "DBindex_cluster_GT":
                        DBindex_list.append(-cluster_GT)
                    elif curriculum == "DBindex_ratio_inst_cluster_GT":
                        DBindex_list.append(inst_DB / cluster_GT)
                    elif curriculum == "DBindex_product_inst_cluster_GT":
                        DBindex_list.append(inst_DB * cluster_GT)
                    elif curriculum == "DBindex_cluster_GT_org_sample_only":
                        DBindex_list.append(-cluster_GT)

                    # DBindex_list.append(metrics.davies_bouldin_score(batched_feature, batched_label))
            if reverse:
                DBid = np.argsort(DBindex_list)[0]
            else:
                DBid = np.argsort(DBindex_list)[len(DBindex_list)-1]
            new_sampled_batch.append(candidate_batch_list[DBid])
            new_sampled_DBindex.append(DBindex_list[DBid])
            if mass_candidate == 'mass_candidate':
                removed_sub = candidate_batch_sub[DBid]
                tobe_batched_idx = np.delete(tobe_batched_idx, removed_sub)
            elif mass_candidate == 'mass_candidate_replacement':
                pass
    print("Curriculum sample_from_mass: resampling DONE.")

    # test_id = np.zeros(1024)
    # for group in new_sampled_batch:
    #     for i in group:
    #         test_id[i] += 1
    # print(new_sampled_batch)
    # print(test_id.astype(np.int).tolist())
    # print(np.sum(test_id))
    # print(new_sampled_DBindex)
    # input()

    return new_sampled_batch

def reorder_DBindex(net, batch_idx_list, data_loader, epoch, use_out=False, reverse=False, half_batch=False, curriculum='', DBindex_use_org_sample=False):
    batch_size = len(batch_idx_list[0])
    if curriculum == '' or curriculum == 'DBindex_high2low':
        batch_DBindex = []
        all_batch_num = len(batch_idx_list)
        if epoch % 2 == 0:
            half_batch_num = all_batch_num // 2
        else:
            half_batch_num = all_batch_num - all_batch_num // 2
        batch_idx_bar = tqdm(batch_idx_list, desc="Reordering")
        for batch_idx in batch_idx_bar:
            pos_1, target = data_loader.get_batch(batch_idx)
            DB, GT_DB = get_batch_DBindex(net, pos_1, target, use_out, DBindex_use_org_sample, curriculum)
            batch_DBindex.append(DB)
        if reverse:
            neworder = np.argsort(batch_DBindex)
            if half_batch:
                neworder = neworder[:half_batch_num]
        else:
            neworder = np.argsort(batch_DBindex)[::-1]
            if half_batch:
                neworder = neworder[:half_batch_num]
        new_batch_idx_list = []
        for i in neworder:
            # print(batch_DBindex[i])
            new_batch_idx_list.append(batch_idx_list[i])

        # print(np.array(batch_DBindex)[neworder])

        # input()

        return new_batch_idx_list

    elif curriculum == "inst_sigma":
        batch_idx_bar = tqdm(batch_idx_list, desc="Reordering")
        inst_sigma = []
        batch_DBindex = []
        for batch_idx in batch_idx_bar:
            pos_1, target = data_loader.get_batch(batch_idx)
            inst_DB, GT_DB = get_batch_DBindex(net, pos_1, target, use_out, DBindex_use_org_sample, curriculum)
            batch_DBindex.append(inst_DB)
            inst_sigma.append(get_inst_sigma(net, pos_1, target, use_out))
        
        inst_sigma = np.concatenate(inst_sigma, axis=0)
        inst_sigma_decrease_idx = np.argsort(-inst_sigma)
        DBindex_decrease_batch = []
        total_num = inst_sigma.shape[0]
        batch_num = (total_num - 1) // batch_size + 1
        for i in range(batch_num):
            DBindex_decrease_batch.append(inst_sigma_decrease_idx[i*batch_size:min(total_num, (i+1)*batch_size)])
        
        batch_idx_bar = tqdm(DBindex_decrease_batch, desc="Reordering inst_sigma_decrease_idx")
        batch_DBindex_decrease = []
        for batch_idx in batch_idx_bar:
            pos_1, target = data_loader.get_batch(batch_idx)
            inst_DB, GT_DB = get_batch_DBindex(net, pos_1, target, use_out)
            batch_DBindex_decrease.append(inst_DB)

    else: # curriculum == "DBindex_ratio_inst_cluster_GT":
        batch_idx_bar = tqdm(batch_idx_list, desc="Reordering")
        batch_inst_DBindex = []
        batch_GT_DBindex = []
        new_batch_idx_list = []
        batch_DBindex_ratio_inst_GT = []
        batch_DBindex_product_inst_GT = []
        for batch_idx in batch_idx_bar:
            pos_1, target = data_loader.get_batch(batch_idx)
            inst_DB, GT_DB = get_batch_DBindex(net, pos_1, target, use_out, DBindex_use_org_sample, curriculum)
            batch_inst_DBindex.append(inst_DB)
            batch_GT_DBindex.append(GT_DB)
            batch_DBindex_ratio_inst_GT.append(inst_DB / GT_DB)
            batch_DBindex_product_inst_GT.append(inst_DB * GT_DB)
        inst_order = np.argsort(batch_inst_DBindex)[::-1]
        if curriculum == "DBindex_ratio_inst_cluster_GT":
            neworder = np.argsort(batch_DBindex_ratio_inst_GT)[::-1]
            # print(np.array(batch_DBindex_ratio_inst_GT)[neworder])
        elif curriculum == "DBindex_cluster_GT":
            neworder = np.argsort(batch_GT_DBindex)
        elif curriculum == "DBindex_cluster_GT_org_sample_only":
            neworder = np.argsort(batch_GT_DBindex)
            # print(np.array(batch_GT_DBindex)[neworder])
        elif curriculum == "DBindex_product_inst_cluster_GT":
            neworder = np.argsort(batch_DBindex_product_inst_GT)[::-1]
            # print(np.array(batch_DBindex_product_inst_GT)[neworder])
        if reverse:
            neworder = neworder[::-1]

        for i in neworder:
            # print(batch_DBindex[i])
            new_batch_idx_list.append(batch_idx_list[i])

        # print(new_batch_idx_list)
        # print(np.array(batch_GT_DBindex)[neworder])
        # input()

        return new_batch_idx_list
        

class ByIndexDataLoader():
    def __init__(self, data_source):
        self.data_source = data_source

    def get_batch(self, indices=None):
        if indices.any() != None:
            batch_data = self.data_source.data[indices].astype(np.float)
            batch_data = torch.tensor(batch_data).permute((0, 3, 1, 2)) / 255.0

            batch_targets = self.data_source.targets[indices]
            if not torch.is_tensor(batch_targets):
                batch_targets = torch.tensor(batch_targets)

            return batch_data.float(), batch_targets

        else:
            raise("Need ordered indices")

    def get_all_data(self):
        all_data = self.data_source.data.astype(np.float)
        all_data = torch.tensor(all_data).permute((0, 3, 1, 2)) / 255.0

        all_targets = self.data_source.targets
        all_targets = torch.tensor(all_targets)

        return all_data.float(), all_targets