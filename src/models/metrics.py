class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum 
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts

class TrainMeter:
    def __init__(self):
        self.loss = 0.
        self.det_loss = 0.
        self.seg_loss = 0.
        self.cam_pos = 0.
        self.cam_neg = 0.
        self.n_counts = 0.

    def add(self, loss, n_counts):
        self.loss += loss
        self.n_counts += n_counts

    def add_all(self,score_list,n_counts):
        # 传入一个字典，返回所有的loss值，用于监控每个loss的情况
        self.loss += score_list["train_loss"]
        self.det_loss += score_list["det_loss"]
        self.seg_loss += score_list["seg_loss"]
        self.cam_pos += score_list["cam_pos"]
        self.cam_neg += score_list["cam_neg"]
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.loss / self.n_counts

    def get_avg_all(self):
        return (self.loss / self.n_counts,
        self.det_loss / self.n_counts,
        self.seg_loss / self.n_counts,
        self.cam_pos / self.n_counts,
        self.cam_neg / self.n_counts)