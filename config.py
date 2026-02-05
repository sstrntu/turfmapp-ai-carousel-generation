import os

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
MAX_SLIDES = int(os.getenv("MAX_SLIDES", "6"))

# ALLOWED_IMAGE_DOMAINS is now optional and defaults to no restrictions (search entire web)
# Set ALLOWED_IMAGE_DOMAINS env var to restrict image sources to specific domains
# Example: ALLOWED_IMAGE_DOMAINS="example.com,other.com"
_domains_env = os.getenv("ALLOWED_IMAGE_DOMAINS", "")
ALLOWED_IMAGE_DOMAINS = [d.strip() for d in _domains_env.split(",") if d.strip()] if _domains_env else None

IMAGE_DOWNLOAD_TIMEOUT = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "12"))

# Text placement rules config file (JSON)
# Set TEXT_PLACEMENT_RULES_PATH to use a custom location
TEXT_PLACEMENT_RULES_FILE = os.getenv("TEXT_PLACEMENT_RULES_FILE", "text_placement_rules.json")
