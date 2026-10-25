"""
Synthetic data generator for PharmaTrack Phase 1.

Generates reproducible synthetic pharmacy data with:
- medicines.csv: medicine_id, medicine_name, category, reorder_level
- inventory.csv: batch_id, medicine_id, quantity, purchase_price, selling_price, expiry_date, supplier
- sales.csv: sale_id, date, batch_id, quantity_sold, unit_price

All data is synthetic with no personal or patient information.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np


class PharmaDataGenerator:
    """Generate synthetic pharmacy data for PharmaTrack."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with fixed random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Synthetic medicine data
        self.medicine_catalog = [
            ("MED001", "Paracetamol 500mg", "Analgesic", 100),
            ("MED002", "Amoxicillin 250mg", "Antibiotic", 50),
            ("MED003", "Ibuprofen 400mg", "Anti-inflammatory", 80),
            ("MED004", "Cetirizine 10mg", "Antihistamine", 60),
            ("MED005", "Omeprazole 20mg", "Antacid", 40),
            ("MED006", "Metformin 500mg", "Antidiabetic", 70),
            ("MED007", "Amlodipine 5mg", "Antihypertensive", 60),
            ("MED008", "Atorvastatin 10mg", "Cholesterol", 50),
            ("MED009", "Salbutamol Inhaler", "Bronchodilator", 30),
            ("MED010", "Vitamin C 1000mg", "Vitamin", 90),
            ("MED011", "Azithromycin 500mg", "Antibiotic", 40),
            ("MED012", "Ranitidine 150mg", "Antacid", 65),
            ("MED013", "Diclofenac 50mg", "Anti-inflammatory", 75),
            ("MED014", "Loratadine 10mg", "Antihistamine", 55),
            ("MED015", "Glimepiride 2mg", "Antidiabetic", 45),
        ]
        
        self.suppliers = [
            "MediSupply Co.",
            "PharmaDistributors Ltd.",
            "HealthCare Wholesale",
            "BioPharm Solutions",
            "MedLine Suppliers",
            "Global Pharma Inc."
        ]
    
    def generate_medicines(self, num_medicines: int = None) -> pd.DataFrame:
        """
        Generate medicines.csv data.
        
        Args:
            num_medicines: Number of medicines to generate (default: all in catalog)
        
        Returns:
            DataFrame with medicine data
        """
        if num_medicines is None:
            catalog = self.medicine_catalog
        else:
            # Use catalog and add synthetic if needed
            catalog = self.medicine_catalog[:num_medicines]
            if len(catalog) < num_medicines:
                for i in range(len(catalog), num_medicines):
                    med_id = f"MED{i+1:03d}"
                    name = f"Synthetic Medicine {i+1}"
                    category = random.choice(["Analgesic", "Antibiotic", "Vitamin", "Antacid"])
                    reorder_level = random.randint(30, 100)
                    catalog.append((med_id, name, category, reorder_level))
        
        data = []
        for med_id, medicine_name, category, reorder_level in catalog:
            data.append({
                "medicine_id": med_id,
                "medicine_name": medicine_name,
                "category": category,
                "reorder_level": reorder_level
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_inventory(self, medicines_df: pd.DataFrame, batches_per_medicine: int = 3) -> pd.DataFrame:
        """
        Generate inventory.csv data with batch information.
        
        Args:
            medicines_df: DataFrame of medicines
            batches_per_medicine: Number of batches per medicine
        
        Returns:
            DataFrame with inventory batch data
        """
        data = []
        batch_counter = 1
        
        for _, med in medicines_df.iterrows():
            medicine_id = med["medicine_id"]
            num_batches = random.randint(2, batches_per_medicine + 1)
            
            for _ in range(num_batches):
                batch_id = f"BATCH{batch_counter:05d}"
                batch_counter += 1
                
                # Generate quantities
                quantity = random.randint(50, 500)
                
                # Generate prices (purchase price lower than selling price)
                purchase_price = round(random.uniform(5.0, 50.0), 2)
                markup = random.uniform(1.2, 2.0)  # 20% to 100% markup
                selling_price = round(purchase_price * markup, 2)
                
                # Generate expiry date (1 month to 2 years from now)
                days_to_expiry = random.randint(30, 730)
                expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")
                
                # Generate supplier
                supplier = random.choice(self.suppliers)
                
                data.append({
                    "batch_id": batch_id,
                    "medicine_id": medicine_id,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "selling_price": selling_price,
                    "expiry_date": expiry_date,
                    "supplier": supplier
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_sales(self, inventory_df: pd.DataFrame, num_sales: int = 200) -> pd.DataFrame:
        """
        Generate sales.csv data.
        
        Args:
            inventory_df: DataFrame of inventory batches
            num_sales: Number of sales transactions
        
        Returns:
            DataFrame with sales data
        """
        data = []
        
        # Get valid batches
        valid_batches = inventory_df[inventory_df['quantity'] > 0].copy()
        
        if valid_batches.empty:
            return pd.DataFrame(columns=["sale_id", "date", "batch_id", "quantity_sold", "unit_price"])
        
        for i in range(num_sales):
            sale_id = f"SALE{i+1:06d}"
            
            # Random batch selection
            batch = valid_batches.sample(1).iloc[0]
            batch_id = batch['batch_id']
            
            # Generate sale date (last 90 days)
            days_ago = random.randint(0, 90)
            sale_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Quantity sold (1 to 20, but not more than available)
            max_qty = min(20, int(batch['quantity']))
            if max_qty < 1:
                continue
            quantity_sold = random.randint(1, max_qty)
            
            # Unit price (slightly different from selling price)
            selling_price = batch['selling_price']
            price_variation = random.uniform(0.9, 1.1)
            unit_price = round(selling_price * price_variation, 2)
            
            data.append({
                "sale_id": sale_id,
                "date": sale_date,
                "batch_id": batch_id,
                "quantity_sold": quantity_sold,
                "unit_price": unit_price
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_all_data(self, 
                         num_medicines: int = 15,
                         batches_per_medicine: int = 3,
                         num_sales: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all three datasets.
        
        Args:
            num_medicines: Number of medicines
            batches_per_medicine: Batches per medicine
            num_sales: Number of sales transactions
        
        Returns:
            Tuple of (medicines_df, inventory_df, sales_df)
        """
        print(f"Generating synthetic data with seed {self.seed}...")
        
        medicines_df = self.generate_medicines(num_medicines)
        print(f"Generated {len(medicines_df)} medicines")
        
        inventory_df = self.generate_inventory(medicines_df, batches_per_medicine)
        print(f"Generated {len(inventory_df)} inventory batches")
        
        sales_df = self.generate_sales(inventory_df, num_sales)
        print(f"Generated {len(sales_df)} sales transactions")
        
        return medicines_df, inventory_df, sales_df
    
    def save_data(self, medicines_df: pd.DataFrame, inventory_df: pd.DataFrame, 
                  sales_df: pd.DataFrame, output_dir: Path = None):
        """
        Save datasets to CSV files.
        
        Args:
            medicines_df: Medicines DataFrame
            inventory_df: Inventory DataFrame
            sales_df: Sales DataFrame
            output_dir: Output directory (default: data/raw)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        medicines_df.to_csv(output_dir / "medicines.csv", index=False)
        inventory_df.to_csv(output_dir / "inventory.csv", index=False)
        sales_df.to_csv(output_dir / "sales.csv", index=False)
        
        print(f"Data saved to {output_dir}")


def generate_sample_data(seed: int = 42, 
                        num_medicines: int = 15,
                        batches_per_medicine: int = 3,
                        num_sales: int = 200,
                        output_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to generate and save sample data.
    
    Args:
        seed: Random seed for reproducibility
        num_medicines: Number of medicines
        batches_per_medicine: Batches per medicine
        num_sales: Number of sales transactions
        output_dir: Output directory
    
    Returns:
        Tuple of DataFrames
    """
    generator = PharmaDataGenerator(seed=seed)
    medicines_df, inventory_df, sales_df = generator.generate_all_data(
        num_medicines, batches_per_medicine, num_sales
    )
    generator.save_data(medicines_df, inventory_df, sales_df, output_dir)
    return medicines_df, inventory_df, sales_df


if __name__ == "__main__":
    generate_sample_data()
