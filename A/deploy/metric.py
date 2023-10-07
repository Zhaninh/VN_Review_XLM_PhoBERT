import numpy as np

class accuracy():
    def __init__(self):
        self.correct = 0
        self.num = 0

    def update(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        self.correct = (y_pred == y_true).sum()
        self.num = len(y_pred)*y_pred.shape[1]

    def compute(self):
        return self.correct/self.num

    def reset(self):
        self.correct = 0
        self.num = 0

def precision(y_pred, y_true):
    epsilon = 1e-7
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_positive = np.logical_and(y_pred, np.logical_not(y_true)).sum(axis=0)
    return (true_positive + epsilon) / (true_positive + false_positive + epsilon)

def recall(y_pred, y_true):
    epsilon = 1e-7
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_negative = np.logical_and(np.logical_not(y_pred), y_true).sum(axis=0)
    return (true_positive + epsilon) / (true_positive + false_negative + epsilon)

class f1_score:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def preprocess_labels(self, y):
        y_np = np.array(y)
        labels = (y_np > 0).astype(int)
        return labels.tolist()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, self.preprocess_labels(y_true)], axis=0) if self.y_true is not None else self.preprocess_labels(y_true)
        self.y_pred = np.concatenate([self.y_pred, self.preprocess_labels(y_pred)], axis=0) if self.y_pred is not None else self.preprocess_labels(y_pred)

    def compute(self):
        self.y_pred = np.array(self.y_pred)
        self.y_true = np.array(self.y_true)
        f1_score = np.zeros(self.y_pred.shape[1])
        precision_score = precision(self.y_pred != 0, self.y_true != 0)
        recall_score = recall(self.y_pred != 0, self.y_true != 0)
        mask_precision_score = np.logical_and(precision_score != 0, np.logical_not(np.isnan(precision_score)))
        mask_recall_score = np.logical_and(recall_score != 0, np.logical_not(np.isnan(recall_score)))
        mask = np.logical_and(mask_precision_score, mask_recall_score)
        print("Precision:",precision_score)
        print("Recall", recall_score)
        f1_score[mask] = 2* (precision_score[mask] * recall_score[mask]) / (precision_score[mask] + recall_score[mask])
        return f1_score

    def reset(self):
        self.y_true = None
        self.y_pred = None

class r2_score:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true], axis=0) if self.y_true is not None else y_true
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0) if self.y_pred is not None else y_pred

    def compute(self):
        mask = np.logical_and(self.y_pred !=0, self.y_true != 0)
        rss = (((self.y_pred - self.y_true)**2)*mask).sum(axis=0) 
        k = (mask*16).sum(axis=0)
        r2_score = np.ones(rss.shape[0])
        mask2 = (k != 0)
        r2_score[mask2] = 1 - rss[mask2]/k[mask2]
        return r2_score

    def reset(self):
        self.y_true = None
        self.y_pred = None

def final_score(f1, r2):
    return (1 / len(f1)) * np.sum(f1 * r2)


if __name__ == '__main__':
    # Khởi tạo y_true và y_pred
    y_true = [[3, 0, 1, 2, 0, 0], [1, 0, 4, 0, 0, 2]]
    y_pred = [[3, 1, 0, 0, 0, 1], [0, 0, 4, 0, 0, 0]]
    y_tconc = [[0, 2, 0, 1, 0, 0], [0, 0, 0, 3, 0, 0]]
    y_pconc = [[0, 2, 0, 0, 2, 0], [0, 2, 0, 3, 0, 0]]

    # 0. Tính accuracy
    ac = accuracy()
    ac.update(y_true, y_pred)
    ac.update(y_tconc, y_pconc)
    acc = ac.compute()
    print(f'Accuracy: {acc}')


    # 1. Tính điểm F1
    f1 = f1_score()
    f1.update(y_true, y_pred)
    f1.update(y_tconc, y_pconc)
    f1_s = f1.compute()
    print(f'F1 Score: {f1_s}')

    # 2. Tính điểm R2
    r2 = r2_score()
    r2.update(y_true, y_pred)
    r2.update(y_tconc, y_pconc)
    r2_s = r2.compute()
    print(f'R2 Score: {r2_s}')


    # 3. Tính final score
    final = final_score(f1_s, r2_s)
    print(f'Final score: {final}')