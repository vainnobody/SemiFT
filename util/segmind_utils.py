import numpy as np
import torch
import torch.nn.functional as F


class MemoryBank:
    def __init__(self, class_num, bank_size, feat_dim):
        self.class_num = class_num
        self.bank_size = bank_size
        self.feat_dim = feat_dim
        self.bank = [[torch.zeros(0, feat_dim)] for _ in range(class_num)]
        self.ptrs = [torch.zeros(1, dtype=torch.long) for _ in range(class_num)]


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    keys_num = keys.shape[0]
    ptr = int(queue_ptr)
    queue[0] = torch.cat((queue[0], keys.detach().cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + keys_num) % queue_size
    queue_ptr[0] = ptr


@torch.no_grad()
def get_mask_tensor(h=512, w=512, mask_gap=16, mask_rate=0.75):
    if h % mask_gap != 0 or w % mask_gap != 0:
        raise ValueError("h and w should be divisible by mask_gap")
    h_gap_num, w_gap_num = int(h / mask_gap), int(w / mask_gap)
    mask_tensor_small = torch.randperm(h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))
    divide_threshold = h_gap_num * w_gap_num * mask_rate
    mask_tensor_small[mask_tensor_small < divide_threshold] = 0.0
    mask_tensor_small[mask_tensor_small >= divide_threshold] = 1.0
    mask_tensor = F.interpolate(mask_tensor_small.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return mask_tensor


@torch.no_grad()
def get_batch_mask_tensor(nchw, mask_gap=16, mask_rate=0.75, device=None):
    mask_tensor = torch.zeros((nchw[0], nchw[-2], nchw[-1]), device=device)
    for img_i in range(nchw[0]):
        mask_tensor[img_i] = get_mask_tensor(h=nchw[-2], w=nchw[-1], mask_gap=mask_gap, mask_rate=mask_rate).to(device)
    return mask_tensor.unsqueeze(1)


@torch.no_grad()
def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)
    if len(labels) == 0:
        return torch.zeros_like(pseudo_labels, dtype=torch.float32)
    labels_select = labels[torch.randperm(len(labels))[: max(len(labels) // 2, 1)]]
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


@torch.no_grad()
def generate_u_data(img_u_w, img_u_s, lab_u, logits_u, entropy_u):
    batch_size = img_u_w.shape[0]
    new_img_w, new_img_s, new_lab, new_logits, new_entropy = [], [], [], [], []
    for i in range(batch_size):
        mix_mask = generate_class_mask(lab_u[i]).to(img_u_w.device)
        next_i = (i + 1) % batch_size
        new_img_w.append((img_u_w[i] * mix_mask + img_u_w[next_i] * (1 - mix_mask)).unsqueeze(0))
        new_img_s.append((img_u_s[i] * mix_mask + img_u_s[next_i] * (1 - mix_mask)).unsqueeze(0))
        new_lab.append((lab_u[i] * mix_mask + lab_u[next_i] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits_u[i] * mix_mask + logits_u[next_i] * (1 - mix_mask)).unsqueeze(0))
        new_entropy.append((entropy_u[i] * mix_mask + entropy_u[next_i] * (1 - mix_mask)).unsqueeze(0))
    return (
        torch.cat(new_img_w),
        torch.cat(new_img_s),
        torch.cat(new_lab).long(),
        torch.cat(new_logits),
        torch.cat(new_entropy),
    )


def contrastive_loss(feat, lab, prob, opt, memory_bank):
    device = feat.device
    loss_c = torch.tensor(0.0, device=device)
    feat_num = feat.shape[1]
    feat = feat.permute(0, 2, 3, 1)

    valid_class_batch_list = []
    feat_mean_batch_list = []
    feat_hard_batch_list = []
    feat_mean_set_tensor = torch.zeros((opt["class_num"], feat_num), device=device)

    for class_i in range(opt["class_num"]):
        lab_i_place = lab == class_i
        if torch.sum(lab_i_place) == 0:
            continue
        valid_class_batch_list.append(class_i)
        prob_i = prob[:, class_i, :, :]
        feat_i_hard_mask = (prob_i < opt["query_threshold"]) * lab_i_place
        dequeue_and_enqueue(
            keys=feat[lab_i_place],
            queue=memory_bank.bank[class_i],
            queue_ptr=memory_bank.ptrs[class_i],
            queue_size=memory_bank.bank_size,
        )
        feat_mean_batch_list.append(torch.mean(feat[lab_i_place], dim=0, keepdim=True))
        feat_hard_batch_list.append(feat[feat_i_hard_mask])
        if len(memory_bank.bank[class_i][0]) > 0:
            feat_mean_set_tensor[class_i] = torch.mean(memory_bank.bank[class_i][0].to(device), dim=0)

    valid_class_num = len(valid_class_batch_list)
    if valid_class_num == 0:
        return loss_c

    for v_class_i, class_kind in enumerate(valid_class_batch_list):
        if len(feat_hard_batch_list[v_class_i]) == 0:
            continue
        feat_hard_idx = torch.randint(len(feat_hard_batch_list[v_class_i]), size=(opt["num_query"],), device=device)
        query_feat = feat_hard_batch_list[v_class_i][feat_hard_idx]

        with torch.no_grad():
            feat_mean_sim = torch.cosine_similarity(feat_mean_batch_list[v_class_i], feat_mean_set_tensor, dim=1)
            feat_mean_sim = torch.cat((feat_mean_sim[:class_kind], feat_mean_sim[class_kind + 1 :]))
            if feat_mean_sim.numel() == 0:
                continue
            negative_sample_prob = torch.softmax(feat_mean_sim, dim=0)
            negative_sample_prob = torch.tensor(
                negative_sample_prob[:class_kind].tolist() + [0.0] + negative_sample_prob[class_kind:].tolist(),
                device=device,
            )
            if negative_sample_prob.sum() == 0:
                continue
            negative_num_sampler = torch.distributions.categorical.Categorical(probs=negative_sample_prob)
            sample_class = negative_num_sampler.sample(sample_shape=[opt["num_query"], opt["num_negative"]])
            sample_class_num = torch.stack([(sample_class == c).sum(1) for c in range(opt["class_num"])], dim=1)
            negative_feat_all = torch.zeros((opt["num_query"], opt["num_negative"], feat_num), device=device)
            for i in range(sample_class_num.shape[0]):
                negative_feat_i_list = []
                for j in range(sample_class_num.shape[1]):
                    memo = memory_bank.bank[j][0]
                    if memo.shape[0] == 0 or int(sample_class_num[i, j]) == 0:
                        continue
                    negative_index = np.random.randint(low=0, high=memo.shape[0], size=int(sample_class_num[i, j])).tolist()
                    negative_feat_i_list.append(memo[negative_index].to(device))
                if negative_feat_i_list:
                    negative_feat_i = torch.cat(negative_feat_i_list)
                    negative_feat_all[i, : negative_feat_i.shape[0]] = negative_feat_i
            positive_feat = feat_mean_batch_list[v_class_i].unsqueeze(0).repeat(opt["num_query"], 1, 1)
            all_feat = torch.cat((positive_feat, negative_feat_all), dim=1)

        seg_logits = torch.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2)
        loss_c = loss_c + F.cross_entropy(seg_logits / opt["temperature"], torch.zeros(opt["num_query"], dtype=torch.long, device=device))
    return loss_c / valid_class_num
