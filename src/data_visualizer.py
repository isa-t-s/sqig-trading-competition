import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


class DataVisualizer:
    """Class to handle data visualization operations"""
    
    def __init__(self, style: str = 'default', figure_size: Tuple[int, int] = (15, 10)):
        """
        Args:
            style: Matplotlib style to use
            figure_size: Default figure size
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figure_size = figure_size
        
    def plot_time_series(self, df: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        title: str = "Time Series Analysis",
                        save_path: Optional[str] = None):
        """
        Plot time series for specified columns
        
        Args:
            df: DataFrame with time series data
            columns: Columns to plot. If None, plots all numeric columns
            title: Main title for the plot
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16)
        
        # Handle single subplot case
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].plot(df.index, df[col], linewidth=1.5)
                axes[i].set_title(f'{col} Over Time')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series plot saved to {save_path}")
        
        plt.show()
    
    def plot_distributions(self, df: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          title: str = "Distribution Analysis",
                          save_path: Optional[str] = None):
        """
        Plot distribution histograms for specified columns
        
        Args:
            df: DataFrame to plot
            columns: Columns to plot. If None, plots all numeric columns
            title: Main title for the plot
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16)
        
        # Handle single subplot case
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = df[col].mean()
                std_val = df[col].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               title: str = "Correlation Matrix",
                               save_path: Optional[str] = None):
        """
        Plot correlation heatmap
        
        Args:
            df: DataFrame to analyze
            columns: Columns to include. If None, uses all numeric columns
            title: Title for the plot
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        correlation_matrix = df[columns].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.show()
        
        return correlation_matrix
    
    def plot_box_plots(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      title: str = "Box Plot Analysis - Outlier Detection",
                      save_path: Optional[str] = None):
        """
        Plot box plots for outlier detection
        
        Args:
            df: DataFrame to plot
            columns: Columns to plot. If None, plots all numeric columns
            title: Main title for the plot
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16)
        
        # Handle single subplot case
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                box_plot = axes[i].boxplot(df[col], patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightblue')
                axes[i].set_title(f'Box Plot of {col}')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
                
                # Add quartile information
                q1, q2, q3 = df[col].quantile([0.25, 0.5, 0.75])
                axes[i].text(1.1, q2, f'Q2: {q2:.2f}', transform=axes[i].get_xaxis_transform())
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Box plots saved to {save_path}")
        
        plt.show()
    
    def plot_scatter_matrix(self, df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           title: str = "Scatter Plot Matrix",
                           save_path: Optional[str] = None):
        """
        Plot scatter matrix for relationship analysis
        
        Args:
            df: DataFrame to plot
            columns: Columns to include. If None, uses all numeric columns
            title: Title for the plot
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) > 6:
            print(f"Warning: Too many columns ({len(columns)}). Using first 6 columns.")
            columns = columns[:6]
        
        from pandas.plotting import scatter_matrix
        
        fig, axes = plt.subplots(figsize=(12, 12))
        scatter_matrix(df[columns], alpha=0.6, figsize=(12, 12), diagonal='hist')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter matrix saved to {save_path}")
        
        plt.show()
    
    def plot_rolling_statistics(self, df: pd.DataFrame,
                               column: str,
                               window: int = 30,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None):
        """
        Plot rolling mean and standard deviation
        
        Args:
            df: DataFrame with time series data
            column: Column to analyze
            window: Rolling window size
            title: Title for the plot
            save_path: Path to save the plot
        """
        if title is None:
            title = f'Rolling Statistics for {column}'
        
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        
        plt.figure(figsize=self.figure_size)
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df[column], label='Original', alpha=0.7)
        plt.plot(df.index, rolling_mean, label=f'{window}-day Rolling Mean', color='red')
        plt.title(f'{column} - Original vs Rolling Mean')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(df.index, rolling_std, label=f'{window}-day Rolling Std', color='orange')
        plt.title(f'{column} - Rolling Standard Deviation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rolling statistics plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   save_dir: Optional[str] = None):
        """
        Create a comprehensive visualization report
        
        Args:
            df: DataFrame to analyze
            columns: Columns to include in analysis
            save_dir: Directory to save plots
        """
        print("=" * 50)
        print("CREATING VISUALIZATION REPORT")
        print("=" * 50)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Time series analysis
        print("1. Creating time series plots...")
        save_path = f"{save_dir}/time_series.png" if save_dir else None
        self.plot_time_series(df, columns, save_path=save_path)
        
        # 2. Distribution analysis
        print("2. Creating distribution plots...")
        save_path = f"{save_dir}/distributions.png" if save_dir else None
        self.plot_distributions(df, columns, save_path=save_path)
        
        # 3. Correlation analysis
        print("3. Creating correlation matrix...")
        save_path = f"{save_dir}/correlation_matrix.png" if save_dir else None
        corr_matrix = self.plot_correlation_matrix(df, columns, save_path=save_path)
        
        # 4. Outlier detection
        print("4. Creating box plots for outlier detection...")
        save_path = f"{save_dir}/box_plots.png" if save_dir else None
        self.plot_box_plots(df, columns, save_path=save_path)
        
        # 5. Rolling statistics for main indicators
        print("5. Creating rolling statistics plots...")
        for col in columns[:2]:  # Limit to first 2 columns to avoid too many plots
            save_path = f"{save_dir}/rolling_{col}.png" if save_dir else None
            self.plot_rolling_statistics(df, col, save_path=save_path)
        
        print("Visualization report completed!")
        
        return corr_matrix


def create_visualization_report(df: pd.DataFrame, 
                              columns: Optional[List[str]] = None) -> DataVisualizer:
    """
    Convenience function to create a comprehensive visualization report
    
    Args:
        df: DataFrame to visualize
        columns: Columns to include in visualization
        
    Returns:
        DataVisualizer instance
    """
    visualizer = DataVisualizer()
    visualizer.create_comprehensive_report(df, columns)
    return visualizer


if __name__ == "__main__":
    print("Data visualizer module loaded successfully!")