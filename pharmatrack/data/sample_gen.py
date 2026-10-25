"""
Synthetic sample data generator for PharmaTrack demonstrations.

Generates clearly labeled synthetic data for:
- medicines.csv (master data)
- batches.csv (batch-level inventory with expiry dates)
- sales.csv (sales transactions referencing batches)

All data is synthetic and contains no real pharmacy or patient information.
"""

from pathlib import Path
from datetime import date, timedelta
import random
import pandas as pd
import numpy as np


# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Synthetic medicine catalog
MEDICINE_CATALOG = [
    # (medicine_id, name, category, manufacturer, unit_selling_price, reorder_level, unit_of_measure)
    ("MED001", "Amoxicillin 500mg", "Antibiotic", "PharmaCorp", 12.50, 50, "capsule"),
    ("MED002", "Paracetamol 500mg", "Analgesic", "MediHealth", 8.00, 100, "tablet"),
    ("MED003", "Ibuprofen 400mg", "NSAID", "PharmaCorp", 10.00, 80, "tablet"),
    ("MED004", "Cetirizine 10mg", "Antihistamine", "AllergyFree", 15.00, 60, "tablet"),
    ("MED005", "Omeprazole 20mg", "PPI", "GastroCare", 18.00, 40, "capsule"),
    ("MED006", "Metformin 500mg", "Antidiabetic", "Diabetix", 22.00, 50, "tablet"),
    ("MED007", "Amlodipine 5mg", "Antihypertensive", "CardioPlus", 16.00, 50, "tablet"),
    ("MED008", "Atorvastatin 20mg", "Statin", "LipidControl", 25.00, 40, "tablet"),
    ("MED009", "Salbutamol Inhaler", "Bronchodilator", "Respire", 45.00, 20, "inhaler"),
    ("MED010", "Vitamin D3 1000IU", "Supplement", "VitaLife", 14.00, 60, "capsule"),
    ("MED011", "Azithromycin 250mg", "Antibiotic", "PharmaCorp", 28.00, 30, "tablet"),
    ("MED012", "Pantoprazole 40mg", "PPI", "GastroCare", 20.00, 40, "tablet"),
    ("MED013", "Losartan 50mg", "Antihypertensive", "CardioPlus", 18.00, 50, "tablet"),
    ("MED014", "Gabapentin 300mg", "Neuropathic", "NeuroMed", 30.00, 30, "capsule"),
    ("MED015", "Levothyroxine 50mcg", "Thyroid", "ThyroCare", 12.00, 50, "tablet"),
]


SUPPLIERS = [
    ("SUP001", "Global Pharma Supply"),
    ("SUP002", "MediDistributors Inc"),
    ("SUP003", "HealthChain Logistics"),
    ("SUP004", "PharmaLink Wholesale"),
    ("SUP005", "BioMed Supply Co"),
]


PAYMENT_METHODS = ["Cash", "Card", "UPI", "Insurance", "Wallet"]


def generate_medicines(n: int = 15) -> pd.DataFrame:
    """Generate medicines master data."""
    rows = []
    for i in range(n):
        if i < len(MEDICINE_CATALOG):
            med = MEDICINE_CATALOG[i]
        else:
            # Generate additional synthetic medicines
            med = (
                f"MED{i+1:03d}",
                f"Synthetic Drug {i+1}",
                random.choice(["Antibiotic", "Analgesic", "NSAID", "Antihistamine", "PPI", "Other"]),
                random.choice(["PharmaCorp", "MediHealth", "GenericLabs", "BioPharma"]),
                round(random.uniform(5.0, 50.0), 2),
                random.randint(20, 100),
                random.choice(["tablet", "capsule", "syrup", "injection", "inhaler"]),
            )
        rows.append({
            "medicine_id": med[0],
            "name": med[1],
            "category": med[2],
            "manufacturer": med[3],
            "unit_selling_price": med[4],
            "reorder_level": med[5],
            "unit_of_measure": med[6],
        })
    return pd.DataFrame(rows)


def generate_batches(medicines: pd.DataFrame, n_batches: int = 60) -> pd.DataFrame:
    """
    Generate batch-level inventory data.

    Each batch has:
    - Unique batch_id
    - medicine_id (FK to medicines)
    - quantity_received, quantity_remaining
    - purchase_cost_per_unit (typically < selling price)
    - manufacture_date, expiry_date
    - supplier_id, supplier_name
    """
    rows = []
    today = date.today()

    for i in range(n_batches):
        med = medicines.sample(1).iloc[0]
        medicine_id = med["medicine_id"]
        selling_price = med["unit_selling_price"]

        # Purchase cost is 40-70% of selling price
        purchase_cost = round(selling_price * random.uniform(0.4, 0.7), 2)

        # Quantity received: 50-500 units
        qty_received = random.randint(50, 500)

        # Some sold/used: remaining is 0-100% of received
        qty_remaining = random.randint(0, qty_received)

        # Manufacture date: 0-365 days ago
        mfg_days_ago = random.randint(0, 365)
        manufacture_date = today - timedelta(days=mfg_days_ago)

        # Expiry: 30-730 days after manufacture (some expired, some near expiry, some fresh)
        # Weight toward creating some expired and near-expiry batches
        expiry_profile = random.choices(
            ["expired", "near_expiry", "normal", "long"],
            weights=[0.15, 0.20, 0.50, 0.15],
            k=1
        )[0]

        if expiry_profile == "expired":
            # Expired: expiry 1-180 days ago
            expiry_days_after_mfg = random.randint(30, 180)
            expiry_date = manufacture_date + timedelta(days=expiry_days_after_mfg)
            # Ensure it's actually expired
            if expiry_date >= today:
                expiry_date = today - timedelta(days=random.randint(1, 30))
        elif expiry_profile == "near_expiry":
            # Near expiry: expires in 1-60 days
            expiry_date = today + timedelta(days=random.randint(1, 60))
        elif expiry_profile == "normal":
            # Normal: expires in 61-365 days
            expiry_date = today + timedelta(days=random.randint(61, 365))
        else:
            # Long: expires in 366-730 days
            expiry_date = today + timedelta(days=random.randint(366, 730))

        supplier_id, supplier_name = random.choice(SUPPLIERS)

        batch_id = f"BATCH{medicine_id[3:]}{i+1:03d}"

        rows.append({
            "batch_id": batch_id,
            "medicine_id": medicine_id,
            "quantity_received": qty_received,
            "quantity_remaining": qty_remaining,
            "purchase_cost_per_unit": purchase_cost,
            "manufacture_date": manufacture_date,
            "expiry_date": expiry_date,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
        })

    df = pd.DataFrame(rows)
    df["manufacture_date"] = pd.to_datetime(df["manufacture_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df


def generate_sales(
    medicines: pd.DataFrame,
    batches: pd.DataFrame,
    n_sales: int = 500,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Generate sales transactions referencing actual batches.

    Each sale references a batch_id and medicine_id for traceability
    and gross profit calculation using batch purchase cost.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=180)
    if end_date is None:
        end_date = date.today()

    # Only use batches that have remaining quantity and are not expired
    saleable_batches = batches[
        (batches["quantity_remaining"] > 0) & (batches["expiry_date"] >= pd.Timestamp(start_date))
    ].copy()

    if saleable_batches.empty:
        # Fallback: use all batches
        saleable_batches = batches.copy()

    rows = []
    for i in range(n_sales):
        # Pick a batch weighted by remaining quantity
        batch = saleable_batches.sample(1, weights="quantity_remaining").iloc[0]
        batch_id = batch["batch_id"]
        medicine_id = batch["medicine_id"]
        purchase_cost = batch["purchase_cost_per_unit"]

        # Get medicine selling price
        med = medicines[medicines["medicine_id"] == medicine_id].iloc[0]
        selling_price = med["unit_selling_price"]

        # Quantity sold: 1-20 units (but not more than remaining)
        max_qty = min(20, batch["quantity_remaining"])
        if max_qty < 1:
            continue
        qty_sold = random.randint(1, max_qty)

        # Sale date within range
        sale_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

        # Selling price can vary slightly (±10%) from master price
        actual_price = round(selling_price * random.uniform(0.9, 1.1), 2)

        payment_method = random.choice(PAYMENT_METHODS)

        sale_id = f"SALE{i+1:06d}"

        rows.append({
            "sale_id": sale_id,
            "batch_id": batch_id,
            "medicine_id": medicine_id,
            "quantity_sold": qty_sold,
            "sale_date": sale_date,
            "selling_price_per_unit": actual_price,
            "payment_method": payment_method,
        })

        # Update remaining quantity in our working copy
        idx = saleable_batches[saleable_batches["batch_id"] == batch_id].index
        if len(idx) > 0:
            saleable_batches.loc[idx, "quantity_remaining"] -= qty_sold

    df = pd.DataFrame(rows)
    if not df.empty:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def generate_sample_data(
    output_dir: Path | None = None,
    n_medicines: int = 15,
    n_batches: int = 60,
    n_sales: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate complete synthetic dataset and save to CSV files.

    Args:
        output_dir: Directory to save CSV files (default: data/sample)
        n_medicines: Number of medicine records
        n_batches: Number of batch records
        n_sales: Number of sales transactions

    Returns:
        Tuple of (medicines, batches, sales) DataFrames.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "sample"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample data in {output_dir}...")

    medicines = generate_medicines(n_medicines)
    medicines.to_csv(output_dir / "medicines.csv", index=False)
    print(f"  Generated {len(medicines)} medicines")

    batches = generate_batches(medicines, n_batches)
    batches.to_csv(output_dir / "batches.csv", index=False)
    print(f"  Generated {len(batches)} batches")

    sales = generate_sales(medicines, batches, n_sales)
    sales.to_csv(output_dir / "sales.csv", index=False)
    print(f"  Generated {len(sales)} sales transactions")

    return medicines, batches, sales


if __name__ == "__main__":
    generate_sample_data()