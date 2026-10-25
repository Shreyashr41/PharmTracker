"""
Inventory Management Module

Provides batch-level inventory tracking, low-stock alerts, 
replenishment suggestions, and reporting functionality.
"""

from .manager import InventoryManager

__all__ = ['InventoryManager']
