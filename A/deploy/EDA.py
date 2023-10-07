import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
from scipy import stats
from preprocessing import preprocess

#-----------------------------------------------------------------------------------------------------------------------

class EDA():
    def __init__(self, dataset):
        self.prep = preprocess()
        self.seg = dataset.map(self.prep.segment)
        self.df = df = pd.DataFrame(self.seg)

    
    def aspect_distribution(self):
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
    
        plt.tight_layout()
        plt.show()

    
    def reviews_len_distribution(self):
        # Tính độ dài của tất cả các review
        review_lengths = self.df['Segment'].str.split().apply(len)
    
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


#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = r"D:\FSoft\Review_Ana\Dream_Tim\A\datasets\data_original\Original-datasets.csv"
    train_df = load_dataset('csv', data_files=data_path)
    eda = EDA(train_df['train'])
    eda.reviews_len_distribution()
    eda.aspect_distribution()