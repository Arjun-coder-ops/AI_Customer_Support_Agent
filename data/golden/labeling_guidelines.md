# Golden Set Labeling Guidelines

## Overview
This document guides human evaluators in reviewing and annotating the golden evaluation dataset (`golden_set.jsonl`) for customer support intent classification.

---

## Instructions

1. Open `data/golden/labeling_candidates.jsonl`.
2. For each record, inspect the `customer_message` and optional `context`.
3. Select exactly ONE intent from the 8 taxonomy classes defined below.
4. Save the verified labels into `data/golden/golden_set.jsonl`.

---

## Intent Taxonomy Summary

| Intent | Description | Key Indicators |
|--------|-------------|----------------|
| `shipping_delay` | Order delayed, tracking stuck | "where is my order", "tracking shows delayed", "late package" |
| `missing_item` | Received parcel missing item(s) | "item missing from box", "incomplete shipment" |
| `order_cancellation` | Request to cancel order before ship | "cancel my order", "ordered by mistake" |
| `refund_return_request` | Return product or money back | "how to return", "refund status", "return label" |
| `account_access_issue` | Login, password, 2FA errors | "cannot login", "reset password", "locked account" |
| `payment_billing_issue` | Double charge, fee, billing error | "charged twice", "billing discrepancy", "payment error" |
| `product_defect_damage` | Broken, defective, damaged item | "damaged box", "item stopped working", "screen cracked" |
| `general_inquiry_feedback` | Restock, features, general questions | "when will this restock", "store hours", "great service" |

---

## Edge Case Rules
- If message expresses both damage AND refund request -> Label `refund_return_request`.
- If message expresses both delayed shipping AND cancel request -> Label `order_cancellation`.
- If intent cannot be determined from text -> Label `general_inquiry_feedback` or flag for escalation.
