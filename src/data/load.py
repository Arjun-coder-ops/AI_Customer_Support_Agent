import os
import logging
import pandas as pd
from typing import Tuple, Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataframe against expected Twitter Customer Support dataset schema.
    Returns (is_valid, error_messages).
    """
    errors = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Data type / structure checks
    if df["tweet_id"].isnull().any():
        errors.append("Found null values in 'tweet_id'")
    
    if df["author_id"].isnull().any():
        errors.append("Found null values in 'author_id'")
        
    if df["inbound"].isnull().any():
        errors.append("Found null values in 'inbound'")

    return len(errors) == 0, errors

def load_dataset(file_path: str, nrows: int = None) -> Tuple[pd.DataFrame, bool]:
    """
    Load raw CSV dataset with schema validation.
    Returns (df, is_real_dataset).
    """
    if not os.path.exists(file_path):
        logger.warning(f"Raw dataset not found at '{file_path}'. Generating clean schema test sample.")
        df = generate_schema_sample()
        return df, False

    logger.info(f"Loading raw dataset from '{file_path}' (nrows={nrows})...")
    df = pd.read_csv(
        file_path,
        nrows=nrows,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": bool,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False,
    )

    is_valid, errors = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed for {file_path}: {errors}")

    logger.info(f"Successfully loaded {len(df)} tweets.")
    return df, True

def generate_schema_sample() -> pd.DataFrame:
    """
    Generate realistic multi-turn Twitter customer support sample dataset
    for testing, schema validation, and offline execution.
    """
    sample_data = [
        # AmazonHelp Conversation 1 (Delayed Shipping)
        {
            "tweet_id": "101",
            "author_id": "cust_101",
            "inbound": True,
            "created_at": "Wed Oct 11 09:00:00 +0000 2017",
            "text": "@AmazonHelp My order #12345 was supposed to arrive yesterday but it shows delayed. Where is it?",
            "response_tweet_id": "102",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "102",
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Oct 11 09:05:00 +0000 2017",
            "text": "@cust_101 We are sorry for the delay! Please DM us your order details and email so we can track it down.",
            "response_tweet_id": "103",
            "in_response_to_tweet_id": "101",
        },
        {
            "tweet_id": "103",
            "author_id": "cust_101",
            "inbound": True,
            "created_at": "Wed Oct 11 09:10:00 +0000 2017",
            "text": "@AmazonHelp Sent you a DM with order #12345. Please check.",
            "response_tweet_id": "104",
            "in_response_to_tweet_id": "102",
        },
        {
            "tweet_id": "104",
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Oct 11 09:20:00 +0000 2017",
            "text": "@cust_101 Thank you! We have updated your tracking info. Your package is out for delivery today.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "103",
        },

        # AmazonHelp Conversation 2 (Missing Item)
        {
            "tweet_id": "201",
            "author_id": "cust_202",
            "inbound": True,
            "created_at": "Wed Oct 11 10:00:00 +0000 2017",
            "text": "@AmazonHelp I received my package today but one of the items is missing from the box!",
            "response_tweet_id": "202",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "202",
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Oct 11 10:04:00 +0000 2017",
            "text": "@cust_202 Oh no! We apologize. Please check your account orders page to see if it shipped in a separate box.",
            "response_tweet_id": "203",
            "in_response_to_tweet_id": "201",
        },
        {
            "tweet_id": "203",
            "author_id": "cust_202",
            "inbound": True,
            "created_at": "Wed Oct 11 10:12:00 +0000 2017",
            "text": "@AmazonHelp It says single shipment in 1 box. Can I get a replacement item sent?",
            "response_tweet_id": "204",
            "in_response_to_tweet_id": "202",
        },
        {
            "tweet_id": "204",
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Oct 11 10:25:00 +0000 2017",
            "text": "@cust_202 Absolutely. We have issued a replacement order free of charge. You will receive email confirmation shortly.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "203",
        },

        # AmazonHelp Conversation 3 (Refund / Return)
        {
            "tweet_id": "301",
            "author_id": "cust_303",
            "inbound": True,
            "created_at": "Wed Oct 11 11:00:00 +0000 2017",
            "text": "@AmazonHelp How do I return a damaged product for a full refund?",
            "response_tweet_id": "302",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "302",
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Oct 11 11:06:00 +0000 2017",
            "text": "@cust_303 You can initiate a return in Your Orders -> Return or Replace Items. Print the label and drop it off.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "301",
        },

        # AppleSupport Conversation 4 (Account Access)
        {
            "tweet_id": "401",
            "author_id": "cust_404",
            "inbound": True,
            "created_at": "Wed Oct 11 12:00:00 +0000 2017",
            "text": "@AppleSupport Locked out of my Apple ID password. Reset link is not coming to email.",
            "response_tweet_id": "402",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "402",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Wed Oct 11 12:05:00 +0000 2017",
            "text": "@cust_404 We can help with Apple ID recovery. Please visit iforgot.apple.com or DM us for support.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "401",
        },

        # Uber_Support Conversation 5 (Payment Issue)
        {
            "tweet_id": "501",
            "author_id": "cust_505",
            "inbound": True,
            "created_at": "Wed Oct 11 13:00:00 +0000 2017",
            "text": "@Uber_Support I was charged twice for my ride this morning. Please refund the duplicate charge.",
            "response_tweet_id": "502",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "502",
            "author_id": "Uber_Support",
            "inbound": False,
            "created_at": "Wed Oct 11 13:08:00 +0000 2017",
            "text": "@cust_505 We would like to look into this right away! Please send us a DM with your phone number and trip date.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "501",
        },
    ]

    df = pd.DataFrame(sample_data)
    return df
