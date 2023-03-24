import pandas as pd
import numpy as np
import os


class Method:
    def __init__(self, data:pd.DataFrame, path_save:str) -> None:
        # Check các cột bắt buộc
        dropped_cols = ["TIME", "PROFIT", "SYMBOL"]
        for col in dropped_cols:
            if col not in data.columns:
                raise Exception(f'Thiếu cột "{col}".')

        # Check kiểu dữ liệu của cột TIME và PROFIT
        if data["TIME"].dtype != "int64":
            raise Exception(f'Kiểu dữ liệu của cột "TIME" phải là int64.')
        if data["PROFIT"].dtype != "float64":
            raise Exception(f'Kiểu dữ liệu của cột "PROFIT" phải là float64.')

        # Check cột TIME xem có tăng dần không
        if data["TIME"].diff().max() > 0:
            raise Exception(f'Dữ liệu phải được sắp xếp giảm dần theo cột "TIME".')

        # Check cột TIME xem có bị khuyết quý nào không
        min_time = np.min(data["TIME"])
        max_time = np.max(data["TIME"])
        time_unique_arr = np.unique(data["TIME"])
        for i in range(min_time, max_time):
            if i not in time_unique_arr:
                raise Exception(f'Dữ liệu bị khuyết chu kỳ "{i}".')

        # Check các cột cần được drop
        for col in data.columns:
            if col not in dropped_cols and data[col].dtype == "object":
                dropped_cols.append(col)

        print(f"Các cột không được coi là biến: {dropped_cols}.")
        self.dropped_cols = dropped_cols

        # Kiểm tra xem path có tồn tại hay không
        if type(path_save) != str or not os.path.exists(path_save):
            raise Exception(f'Không tồn tại thư mục {path_save}/.')
        else:
            if not path_save.endswith("/") and not path_save.endswith("\\"):
                path_save += "/"

            self.path = path_save

        self.TRAINING_DATA = data
        self.PROFIT = np.array(self.TRAINING_DATA["PROFIT"], dtype=np.float64)
        self.OPERAND = np.transpose(np.array(self.TRAINING_DATA.drop(columns=dropped_cols), dtype=np.float64))
        time_arr = np.array(self.TRAINING_DATA["TIME"])
        qArr = np.unique(time_arr)
        self.INDEX = np.full(qArr.shape[0]+1, 0, dtype=np.int64)
        for i in range(qArr.shape[0]):
            if i == qArr.shape[0] - 1:
                self.INDEX[qArr.shape[0]] = time_arr.shape[0]
            else:
                temp = time_arr[self.INDEX[i]]
                for j in range(self.INDEX[i], time_arr.shape[0]):
                    if time_arr[j] != temp:
                        self.INDEX[i+1] = j
                        break
