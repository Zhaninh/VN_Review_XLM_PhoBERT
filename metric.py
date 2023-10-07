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

    def calc(self):
        return self.correct/self.num

    def reset(self):
        self.correct = 0
        self.num = 0

class f1:
    def __init__(self):
        self.num_classes = 6
        self.epsilon = 1e-7

    def calc(self, y_true, y_pred):
        f1_scores = []
        for class_idx in range(self.num_classes):
            true_positive = np.sum((y_true == class_idx) & (y_pred == class_idx))
            false_positive = np.sum((y_true != class_idx) & (y_pred == class_idx))
            false_negative = np.sum((y_true == class_idx) & (y_pred != class_idx))

            precision = true_positive / (true_positive + false_positive + self.epsilon)
            recall = true_positive / (true_positive + false_negative + self.epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

            f1_scores.append(f1)

        average_f1_score = np.mean(f1_scores)
        return average_f1_score

class r2:
    def calc(self, y_true, y_pred):
        mean_y_true = np.mean(y_true)
        total_sum_squares = np.sum((y_true - mean_y_true) ** 2)
        residual_sum_squares = np.sum((y_true - y_pred) ** 2)
        r2_score = 1 - (residual_sum_squares / (total_sum_squares + 1e-7))
        return r2_score

class cm:
    def __init__(self):
        self.num_classes = 6

    def calc(self, y_true, y_pred):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                confusion_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
        return confusion_matrix


if __name__ == "__main__":
    # Khởi tạo y_true và y_pred
    y_true = np.array([[3, 0, 0, 2, 0, 0], [0, 0, 4, 0, 0, 2]])
    y_pred = np.array([[3, 1, 0, 2, 0, 0], [0, 0, 4, 0, 0, 3]])

    # 1. Tính điểm F1
    f1_calculator = f1()
    average_f1_score = f1_calculator.calc(y_true, y_pred)
    print(f'Average F1 Score: {average_f1_score}')

    # 2. Tính điểm R2
    r2_calculator = r2()
    r2_score = r2_calculator.calc(y_true.flatten(), y_pred.flatten())
    print(f'R2 Score: {r2_score}')

    # 3. Tính confusion matrix
    confusion_matrix_calculator = cm()
    confusion_matrix = confusion_matrix_calculator.calc(y_true.flatten(), y_pred.flatten())
    print('Confusion Matrix:')
    print(confusion_matrix)
