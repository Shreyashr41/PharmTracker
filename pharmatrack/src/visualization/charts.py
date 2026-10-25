"""
PharmaTrack Visualization Functions

Provides reusable chart functions using Matplotlib and Seaborn.
All visualization logic is separated from page code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Tuple, List


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class PharmaCharts:
    """Reusable chart functions for PharmaTrack dashboard."""
    
    @staticmethod
    def line_chart_time_series(df: pd.DataFrame, x_col: str, y_col: str, 
                               title: str, xlabel: str, ylabel: str,
                               figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Create time-series line chart using Matplotlib.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return fig
        
        # Handle date formatting
        df_plot = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            ax.plot(df_plot[x_col], df_plot[y_col], marker='o', linewidth=2, markersize=6)
            fig.autofmt_xdate()
        else:
            ax.plot(df_plot[x_col], df_plot[y_col], marker='o', linewidth=2, markersize=6)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
                  title: str, xlabel: str, ylabel: str,
                  figsize: Tuple = (10, 6), horizontal: bool = False) -> plt.Figure:
        """
        Create bar chart using Matplotlib.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return fig
        
        df_plot = df.copy().head(15)  # Limit to top 15 for readability
        
        if horizontal:
            ax.barh(df_plot[x_col], df_plot[y_col], color=sns.color_palette("viridis", len(df_plot)))
            ax.set_xlabel(ylabel, fontsize=11)
            ax.set_ylabel(xlabel, fontsize=11)
        else:
            ax.bar(df_plot[x_col], df_plot[y_col], color=sns.color_palette("viridis", len(df_plot)))
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap",
                           figsize: Tuple = (10, 8)) -> plt.Figure:
        """
        Create correlation heatmap using Seaborn.
        Only shows meaningful correlations when sufficient numerical columns exist.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select only numerical columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            ax.text(0.5, 0.5, 'Insufficient numerical columns for correlation heatmap\n(Need at least 2 numeric columns)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            return fig
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    fmt='.2f', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def box_plot(df: pd.DataFrame, x_col: str = None, y_col: str = None,
                 title: str = "Box Plot", figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Create box plot using Seaborn for statistical distribution.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if x_col and y_col and x_col in df.columns and y_col in numeric_df.columns:
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette='Set2')
        elif y_col and y_col in numeric_df.columns:
            sns.boxplot(data=df, y=y_col, ax=ax, palette='Set2')
        elif len(numeric_df.columns) > 0:
            sns.boxplot(data=numeric_df, ax=ax, palette='Set2')
        else:
            ax.text(0.5, 0.5, 'No numerical data for box plot', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def distribution_chart(df: pd.DataFrame, column: str, 
                          title: str = "Distribution", bins: int = 20,
                          figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Create statistical distribution chart using Seaborn.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if df.empty or column not in df.columns:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        numeric_series = df[column].dropna()
        if numeric_series.empty or not pd.api.types.is_numeric_dtype(numeric_series):
            ax.text(0.5, 0.5, 'No numerical data for distribution', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        sns.histplot(data=numeric_series, bins=bins, kde=True, ax=ax, color='skyblue')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def category_bar_chart(df: pd.DataFrame, category_col: str, value_col: str,
                          title: str, xlabel: str, ylabel: str,
                          figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Create bar chart for category comparisons.
        """
        return PharmaCharts.bar_chart(df, category_col, value_col, title, xlabel, ylabel, figsize)
    
    @staticmethod
    def pie_chart(df: pd.DataFrame, labels_col: str, values_col: str,
                  title: str, figsize: Tuple = (10, 8)) -> plt.Figure:
        """
        Create pie chart for distribution.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if df.empty or labels_col not in df.columns or values_col not in df.columns:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        df_plot = df.copy().head(8)  # Limit to top 8 for readability
        ax.pie(df_plot[values_col], labels=df_plot[labels_col], autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette("Set3", len(df_plot)))
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
