#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
from scipy import stats
import nbimporter
from preprocessing import preprocess


# In[2]:


class EDA():
    def __init__(self, dataset):
        self.prep = preprocess()
        self.df = dataset

    def count_label_distribution(self, y):
        label_counts = pd.DataFrame(data=y, columns=['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam'])
        label_distribution = label_counts.apply(lambda x: x.value_counts()).fillna(0).astype(int)
        return label_distribution

    
    def aspect_distribution(self):
        self.df = pd.DataFrame(self.df)
        # Danh sách các khía cạnh
        aspects = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']
    
        # Vẽ biểu đồ histogram cho từng khía cạnh
        plt.figure(figsize=(10, 6))
        for i, aspect in enumerate(aspects):
            plt.subplot(2, 3, i+1)
            plt.hist(self.df[aspect], bins=10, alpha=0.7, color='b', edgecolor='black')
            plt.title(f'Phân phối của {aspect}')
            plt.xlabel(aspect)
            plt.ylabel('Số lượng')
            plt.grid(axis='y', alpha=0.75)
        plt.subplots_adjust(wspace=0.6, hspace=0.5)  # Điều chỉnh khoảng cách giữa các biểu đồ

        # Tạo DataFrame để lưu trữ dữ liệu
        data = {}
        for aspect in aspects:
            aspect_data = {f'{j} Sao': (self.df[aspect] == j).sum() for j in range(6)}
            data[aspect] = aspect_data

        data_df = pd.DataFrame(data)

        # Tạo list cho cột "Đánh giá" với các giá trị từ 0 đến 5
        danh_gia = ['0', '1', '2', '3', '4', '5']
        
        # Thêm cột "Đánh giá" phía trước DataFrame
        data_df.insert(0, 'Đánh giá', danh_gia)
        
        # In DataFrame ra và hiển thị
        print(data_df.to_string(index=False))
        
        plt.tight_layout()
        plt.show()


    

    
    def reviews_len_distribution(self):
        seg = self.df.map(self.prep.segment)
        seg_df = pd.DataFrame(seg)
        
        # Tính độ dài của tất cả các review
        review_lengths = seg_df['Segment'].str.split().apply(len)
    
        # Tạo biểu đồ histogram
        plt.figure(figsize=(10, 5))
        n, bins, _ = plt.hist(review_lengths, bins=50, density=True, color='b', edgecolor='black', alpha=0.7)
    
        # Tính và vẽ đường kernel density estimation (KDE)
        kernel = stats.gaussian_kde(review_lengths)
        kernel.covariance_factor = lambda : .25
        kernel._compute_covariance()
        kernel_values = kernel(bins)
        plt.plot(bins, kernel_values, 'r-', linewidth=2)
    
        plt.title('Phân phối độ dài của các review')
        plt.xlabel('Độ dài review')
        plt.ylabel('Tần số')
    
        # Tính các thông số thống kê
        mean_length = review_lengths.mean()
        std_length = review_lengths.std()
        q1_length = review_lengths.quantile(0.25)
        q2_length = review_lengths.quantile(0.50)
        q3_length = review_lengths.quantile(0.75)
        max_length = review_lengths.max()
        min_length = review_lengths.min()
    
        # Hiển thị các giá trị thống kê bên ngoài biểu đồ
        stats_text = f"Mean: {mean_length:.2f}\n"
        stats_text += f"Standard Deviation: {std_length:.2f}\n"
        stats_text += f"Q1: {q1_length}\n"
        stats_text += f"Q2 (Median): {q2_length}\n"
        stats_text += f"Q3: {q3_length}\n"
        stats_text += f"Max: {max_length}\n"
        stats_text += f"Min: {min_length}"
    
        plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=12)
    
        plt.grid(axis='y', alpha=0.75)
        plt.show()



