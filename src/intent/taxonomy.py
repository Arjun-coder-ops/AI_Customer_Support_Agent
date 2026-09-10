import os
import yaml
from typing import Dict, Any, List

class IntentTaxonomy:
    """
    Manages loading, validating, and retrieving intent taxonomy definitions.
    """
    def __init__(self, taxonomy_path: str = "data/golden/taxonomy.yaml"):
        self.taxonomy_path = taxonomy_path
        self.intents = []
        self.intent_names = []
        self.intent_map = {}
        self.load_taxonomy()

    def load_taxonomy(self):
        if not os.path.exists(self.taxonomy_path):
            raise FileNotFoundError(f"Taxonomy file not found at '{self.taxonomy_path}'")
        
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.intents = data.get("intents", [])
        self.intent_names = [i["name"] for i in self.intents]
        self.intent_map = {i["name"]: i for i in self.intents}

    def get_intent_names(self) -> List[str]:
        return self.intent_names

    def get_intent_info(self, intent_name: str) -> Dict[str, Any]:
        return self.intent_map.get(intent_name, {})

    def get_all_intents(self) -> List[Dict[str, Any]]:
        return self.intents
