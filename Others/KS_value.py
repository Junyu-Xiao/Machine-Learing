import numpy as np

def TPR(goal, kind, t, k, num):

    if kind == goal:
        return k/num
    else:
        return (t-k)/num


def KS_value(label, pred, label_kind):
    ##计算KS值

    ##格式转化
    label = np.array(label).astype('int64')
    pred = np.array(pred).astype('float32')
    label_kind = int(label_kind)

    ##统计数量
    new_label = np.where(label == label_kind, 0, 1)
    num_0 = np.bincount(new_label)[0]
    num_1 = np.bincount(new_label)[1]

    ##排序
    tsort_index = np.argsort(pred)[::-1]
    tsort = label.copy()
    tsort[tsort_index] = np.arange(1, tsort_index.shape[0] + 1)

    ksort = label.copy()
    kind_0_index = np.where(new_label == 0)[0]
    kind_1_index = np.where(new_label == 1)[0]
    kind_0_sort = np.argsort(pred[kind_0_index])[::-1] 
    kind_1_sort = np.argsort(pred[kind_1_index])[::-1] 
    ksort[kind_0_index[kind_0_sort]] = np.arange(1, kind_0_sort.shape[0] + 1)
    ksort[kind_1_index[kind_1_sort]] = np.arange(1, kind_1_sort.shape[0] + 1)

    ##计算
    ks_arr = np.vstack([new_label, tsort, ksort]).T[tsort_index]
    new_label = ks_arr[:, 0]
    tsort = ks_arr[:, 1]
    ksort = ks_arr[:, 2]
    tpr = np.vectorize(TPR)(0, new_label, tsort, ksort, num_0)
    fpr = np.vectorize(TPR)(1, new_label, tsort, ksort, num_1)
    ks = np.vectorize(lambda x, y: abs(x - y))(tpr, fpr)
    return np.max(ks)
