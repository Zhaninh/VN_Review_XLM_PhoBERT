import torch
import torch.nn as nn
import torch.nn.functional as F



def SigmoidFocalLoss(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = 'none'):

    '''
    - inputs, labels = [0|1]
    
    - alpha:
    siêu tham số khác quyết định mức độ cân bằng giữa các lớp trong trường hợp mất cân bằng dữ liệu. 
    Nếu bạn muốn tập trung nhiều hơn vào lớp thiểu số, bạn có thể đặt giá trị alpha cao hơn cho lớp thiểu số.

    - gamma:
    siêu tham số quyết định mức độ tập trung của hàm mất mát vào các ví dụ khó phân loại. 
    Giá trị gamma cao hơn đặt nhiều trọng số hơn cho các ví dụ bị phân loại sai. Thường thì giá trị gamma được đặt trong khoảng từ 1 đến 5.

    - reduction:
    Tham số này xác định cách hàm mất mát được thu gọn qua toàn bộ mẫu dữ liệu. 
    Nó có thể có giá trị "none" (không thu gọn), "mean" (trung bình của các mất mát), hoặc "sum" (tổng của các mất mát).
    '''

    p = inputs # 0 or 1
    bce_loss = F.binary_cross_entropy(inputs, labels, reduction="none")
    
    p_t = p * labels + (1 - p) * (1 - labels) 
    # nếu p và labels (0 or 1) trừng nhau --> dự đoán đúng --> p_t = 1
    # nếu p và labels (0 or 1) khác nhau --> dự đoán sai --> p_t = 0
    
    loss = bce_loss * ((1 - p_t) ** gamma)
    '''
    - loss có cùng hình dạng và kích thước với ce_loss. 
    Nó chứa các giá trị mất mát focal loss đã được điều chỉnh để tập trung vào các ví dụ khó phân loại.
    '''

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss



def loss_classifier(pred_classifier, labels_classifier):
    loss_fn = nn.BCELoss()
    return loss_fn(pred_classifier, labels_classifier)



def loss_regressor(pred_regressor, labels_regressor):
    mask = (labels_regressor != 0)
    loss = ((pred_regressor - labels_regressor)**2)[mask].sum() / mask.sum()
    return loss



def loss_softmax(inputs, labels, device):
    mask = (labels != 0)
    n, aspect, rate = inputs.shape
    num = 0
    loss = torch.zeros(labels.shape).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    for i in range(aspect):
        label_i = labels[:, i].clone()
        label_i[label_i != 0] -= 1
        label_i = label_i.type(torch.LongTensor).to(device)
        loss[:, i] = loss_fn(inputs[:, i, :], label_i)
    loss = loss[mask].sum() / mask.sum()
    return loss
