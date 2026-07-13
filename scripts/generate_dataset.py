"""Generate a realistic synthetic churn dataset with controlled noise."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

ROWS = 500
SEED = 42


def generate_row(rng: random.Random) -> dict:
    """Generate one synthetic customer row with realistic churn signals."""

    contract = rng.choices(
        ["month_to_month", "annual", "two_year"],
        weights=[0.50, 0.30, 0.20],
    )[0]
    payment = rng.choices(
        ["credit_card", "bank_transfer", "invoice"],
        weights=[0.45, 0.35, 0.20],
    )[0]
    tier = rng.choices(
        ["starter", "standard", "enterprise"],
        weights=[0.30, 0.45, 0.25],
    )[0]

    # Tenure correlates with contract type
    if contract == "month_to_month":
        tenure = rng.randint(1, 24)
    elif contract == "annual":
        tenure = rng.randint(6, 60)
    else:
        tenure = rng.randint(12, 84)

    # Charges correlate with tier
    base_charge = {"starter": 69, "standard": 129, "enterprise": 199}[tier]
    monthly_charges = round(base_charge + rng.gauss(0, 20), 2)
    monthly_charges = max(29.0, monthly_charges)
    total_charges = round(monthly_charges * tenure + rng.gauss(0, 100), 2)
    total_charges = max(monthly_charges, total_charges)

    seats = {"starter": rng.randint(1, 10),
             "standard": rng.randint(5, 50),
             "enterprise": rng.randint(20, 100)}[tier]

    has_auto_pay = rng.random() < (0.3 if contract == "month_to_month" else 0.7)

    # Support tickets: higher for churners, but with overlap
    base_tickets = rng.randint(0, 3)
    support_tickets = base_tickets + (rng.randint(0, 5) if contract == "month_to_month" else 0)
    support_tickets = min(support_tickets, 20)

    # --- Churn probability model ---
    # Short tenure, month-to-month, no autopay, high tickets → higher churn
    churn_score = 0.0

    # Contract is the strongest signal
    churn_score += {"month_to_month": 0.35, "annual": 0.10, "two_year": 0.02}[contract]

    # Tenure: shorter = riskier
    if tenure <= 6:
        churn_score += 0.20
    elif tenure <= 12:
        churn_score += 0.10
    elif tenure >= 36:
        churn_score -= 0.10

    # Auto-pay is a retention signal
    if not has_auto_pay:
        churn_score += 0.10

    # Support tickets
    if support_tickets >= 5:
        churn_score += 0.15
    elif support_tickets >= 3:
        churn_score += 0.05

    # Tier: enterprise customers have higher switching costs
    churn_score += {"starter": 0.05, "standard": 0.0, "enterprise": -0.05}[tier]

    # Add noise so the model can't be perfect
    churn_score += rng.gauss(0, 0.12)
    churn_prob = max(0.0, min(1.0, churn_score))

    churned = 1 if rng.random() < churn_prob else 0

    return {
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets_last_90d": support_tickets,
        "contract_type": contract,
        "payment_method": payment,
        "has_auto_pay": str(has_auto_pay).lower(),
        "product_tier": tier,
        "seats": seats,
        "churned": churned,
    }


def main() -> None:
    rng = random.Random(SEED)
    output_path = PROJECT_ROOT / "data" / "customer_churn.csv"

    fieldnames = FEATURE_COLUMNS + [TARGET_COLUMN]
    rows = [generate_row(rng) for _ in range(ROWS)]

    # Report class balance
    churn_count = sum(1 for r in rows if r["churned"] == 1)
    retain_count = ROWS - churn_count
    print(f"Generated {ROWS} rows: {churn_count} churn, {retain_count} retain")
    print(f"Churn rate: {churn_count / ROWS:.1%}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
