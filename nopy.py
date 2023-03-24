from numba import njit
import numpy as np


@njit
def get_valid_op(struct, idx, start):
    valid_op = np.full(2, 0)
    valid_op[start-2:] = 1

    if idx // 2 <= struct[0,1] // 2:
        valid_op[1] = 0

    return np.where(valid_op == 1)[0] + 2


@njit
def get_valid_operand(formula, struct, idx, start, num_operand):
    valid_operand = np.full(num_operand, 0)
    valid_operand[start:num_operand] = 1

    for i in range(struct.shape[0]):
        if struct[i,2] + 2*struct[i,1] > idx:
            gr_idx = i
            break

    """
    Tránh hoán vị nhân chia trong một cụm
    """
    pre_op = formula[idx-1]
    if pre_op >= 2:
        if pre_op == 2:
            temp_idx = struct[gr_idx,2]
            if idx >= temp_idx + 2:
                valid_operand[0:formula[idx-2]] = 0
        else:
            temp_idx = struct[gr_idx,2]
            temp_idx_1 = temp_idx + 2*struct[gr_idx,3]
            if idx > temp_idx_1 + 2:
                valid_operand[0:formula[idx-2]] = 0

            """
            Tránh chia lại những toán hạng đã nhân ở trong cụm (chỉ phép chia mới check)
            """
            valid_operand[formula[temp_idx:temp_idx_1+1:2]] = 0

    """
    Tránh hoán vị cộng trừ các cụm, kể từ cụm thứ 2 trở đi
    """
    if gr_idx > 0:
        gr_check_idx = -1
        for i in range(gr_idx-1,-1,-1):
            if struct[i,0]==struct[gr_idx,0] and struct[i,1]==struct[gr_idx,1] and struct[i,3]==struct[gr_idx,3]:
                gr_check_idx = i
                break

        if gr_check_idx != -1:
            idx_ = 0
            while True:
                idx_1 = struct[gr_idx,2] + idx_
                idx_2 = struct[gr_check_idx,2] + idx_
                if idx_1 == idx:
                    valid_operand[0:formula[idx_2]] = 0
                    break

                if formula[idx_1] != formula[idx_2]:
                    break

                idx_ += 2

        """
        Tránh trừ đi những cụm đã cộng trước đó (chỉ ở trong trừ cụm mới check)
        """
        if struct[gr_idx,0] == 1 and idx + 2 == struct[gr_idx,2] + 2*struct[gr_idx,1]:
            list_gr_check = np.where((struct[:,0]==0) & (struct[:,1]==struct[gr_idx,1]) & (struct[:,3]==struct[gr_idx,3]))[0]
            for i in list_gr_check:
                temp_idx = struct[i,2] + 2*struct[i,1] - 2
                temp_idx_1 = struct[gr_idx,2] + 2*struct[gr_idx,1] - 2
                if (formula[struct[i,2]:temp_idx] == formula[struct[gr_idx,2]:temp_idx_1]).all():
                    valid_operand[formula[temp_idx]] = 0

    return np.where(valid_operand==1)[0]


@njit
def get_value_index_profit(weight, profit, index):
    size = index.shape[0]-1
    arr_value = np.zeros(size, dtype=np.float64)
    arr_index = np.zeros(size, dtype=np.int64)
    arr_profit = np.zeros(size, dtype=np.float64)

    for i in range(index.shape[0]-2, -1, -1):
        idx = index.shape[0]-2-i
        temp = weight[index[i]:index[i+1]]
        max_ = np.max(temp)
        max_idx = np.where(temp == max_)[0] + index[i]
        if max_idx.shape[0] == 1:
            arr_value[idx] = max_
            arr_index[idx] = max_idx[0]
            arr_profit[idx] = profit[max_idx[0]]
            if profit[max_idx[0]] <= 0.0:
                return arr_value, arr_index, arr_profit
        else:
            arr_value[idx] = max_
            arr_index[idx] = -1
            arr_profit[idx] = 1.0

    return arr_value, arr_index, arr_profit


@njit
def geo_geo_L_value_geo_L(value, index, profit, target):
    geo = np.prod(profit) ** (1.0/profit.shape[0])
    if geo < target:
        return 0.0, 0.0, 0.0

    min_value = np.min(value)
    value_geo_L = min_value - np.max(np.array([1e-6, 1e-6*np.abs(min_value)]))
    geo_L = geo
    temp_value = value.copy()
    temp_value[np.where(index==-1)[0]] = 1.7976931348623157e+308
    for x in temp_value:
        temp_profit = 1.0
        for i in range(temp_value.shape[0]):
            if temp_value[i] > x:
                temp_profit *= profit[i]
            else:
                temp_profit *= 1.01

        geo_ = temp_profit ** (1.0/temp_value.shape[0])
        if geo_ > geo_L:
            geo_L = geo_
            value_geo_L = x

    return geo, geo_L, value_geo_L


@njit
def har_har_L_value_har_L(value, index, profit, target):
    temp = 0.0
    for p in profit:
        if p <= 0.0:
            temp = 1.7976931348623157e+308
            break

        temp += 1.0/p

    har = profit.shape[0] / temp
    if har < target:
        return 0.0, 0.0, 0.0

    min_value = np.min(value)
    value_har_L = min_value - np.max(np.array([1e-6, 1e-6*np.abs(min_value)]))
    har_L = har
    temp_value = value.copy()
    temp_value[np.where(index==-1)[0]] = 1.7976931348623157e+308
    for x in temp_value:
        temp = 0.0
        for i in range(temp_value.shape[0]):
            if temp_value[i] > x:
                temp += 1.0/profit[i]
            else:
                temp += 1.0/1.01

        har_ = profit.shape[0] / temp
        if har_ > har_L:
            har_L = har_
            value_har_L = x

    return har, har_L, value_har_L


@njit
def get_bit_mean(weight, profit):
    com_los = np.where(profit < 1.0)[0]
    com_win = np.where(profit > 1.0)[0]
    max_los = np.max(weight[com_los])
    a = np.count_nonzero(weight[com_win] < max_los)
    return 1 - (a/com_win.shape[0])
