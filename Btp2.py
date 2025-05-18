# General Imports
import hashlib
import time
import json
import base64
import multiprocessing
import logging
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
import os
import threading
import sqlite3
from pathlib import Path
import configparser
import argparse
import random # For simulating fraud scores
import pandas as pd # For loading the CSV dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sys # To check for interactive environment and for package installation

# --- Tabulate Handling ---
# Attempt to import tabulate, fallback to a manual version
try:
    import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    print("Installing 'tabulate' library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabulate"])
    try:
        import tabulate
        TABULATE_AVAILABLE = True
        print("Successfully installed 'tabulate'.")
    except ImportError:
        TABULATE_AVAILABLE = False
        print("Failed to install 'tabulate'. Manual table formatting will be used.")

# Cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature

# IPython imports (for Colab/Jupyter)
try:
    from IPython import get_ipython
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# --- Setup Logging ---
# Configure logging to avoid duplicate handlers if run multiple times in a notebook
def setup_logging():
    # Get the root logger and check for existing handlers
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    else:
        # If handlers exist, assume basicConfig was already called.
        # We might need to adjust levels or add specific handlers if needed.
        pass

    # Configure the specific 'blockchain' logger, preventing propagation to root
    logger = logging.getLogger("blockchain")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False # Prevent propagation to root

    # Configure the logger for tabulate fallback if needed
    if not TABULATE_AVAILABLE:
        logger_fallback_tabulate = logging.getLogger("blockchain_tabulate_fallback")
        if not logger_fallback_tabulate.handlers:
            logger_fallback_tabulate.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger_fallback_tabulate.addHandler(handler)
            logger_fallback_tabulate.propagate = False

# Call setup_logging early
setup_logging()
logger = logging.getLogger("blockchain") # Re-get logger after setup

# --- Load Configuration ---
config = configparser.ConfigParser()
# Define a default config path relative to the script if possible, or use current working directory
try:
    # This works in standard Python scripts
    _script_dir = Path(__file__).resolve().parent
except NameError:
    # This works in interactive environments like Jupyter/Colab
    _script_dir = Path.cwd()

config_file_name = "blockchain_config.ini"
config_path = _script_dir / config_file_name

# Try to read config, if fails or not exists, create with defaults
if config_path.exists():
    try:
        config.read(config_path)
        logger.info(f"Loaded config from {config_path}")
    except Exception as e:
        logger.warning(f"Could not read config file {config_path}: {e}. Using default values.")
        # Proceed to create default config below if reading fails
        config.clear() # Clear any partial read
else:
    logger.info(f"Config file not found at {config_path}. Creating default config.")

# Apply default configuration values if sections/keys are missing
if not config.has_section("BLOCKCHAIN"):
    config.add_section("BLOCKCHAIN")
config_defaults = {
    "difficulty": "3",
    "max_transactions_per_block": "5",
    "block_time_target_seconds": "15",
    "difficulty_adjustment_interval": "5"
}
for key, value in config_defaults.items():
    if not config.has_option("BLOCKCHAIN", key):
        config.set("BLOCKCHAIN", key, value)

if not config.has_section("DATABASE"):
     config.add_section("DATABASE")
if not config.has_option("DATABASE", "path"):
    config.set("DATABASE", "path", "blockchain.db") # Default database file path

# Try to write the config file back with defaults if it didn't exist or had missing sections
if not config_path.exists() or not all(config.has_section(s) and all(config.has_option(s, k) for k in config_defaults) for s in ["BLOCKCHAIN"]):
    try:
        with open(config_path, "w") as f:
            config.write(f)
            logger.info(f"Created/Updated config file at {config_path} with default values.")
    except Exception as e:
        logger.warning(f"Could not write default config file to {config_path}: {e}.")


# ============================
# Cryptography & Hashing
# ============================
def calculate_sha256(data: str, salt: str = "") -> str:
    """Calculates the SHA256 hash of input data with an optional salt."""
    return hashlib.sha256((str(data) + salt).encode()).hexdigest()

def hash_dict(data: Dict) -> str:
    """Calculates the SHA256 hash of a dictionary by serializing it consistently."""
    # Ensure consistent hashing by sorting keys before dumping to JSON
    ordered_data = OrderedDict(sorted(data.items()))
    return calculate_sha256(json.dumps(ordered_data))

class Wallet:
    """Represents a user's wallet with public and private keys."""
    def __init__(self, load_path: Optional[str] = None):
        # Key loading from file is not implemented in this demo.
        # New keys are always generated for simplicity.
        self._generate_keys()
        if load_path:
            logger.debug(f"Wallet key loading from path '{load_path}' is disabled in this demo. New keys generated.")

    def _generate_keys(self):
        """Generates a new RSA public and private key pair."""
        # Use a larger key size for better security in a real application
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        logger.debug("New wallet keys generated")

    def sign_transaction(self, transaction_data: Dict) -> str:
        """Signs a transaction dictionary using the wallet's private key."""
        # Sign an ordered dictionary of the transaction data to ensure consistent signing
        ordered_data_for_signing = OrderedDict(sorted(transaction_data.items()))
        # Ensure the data is in bytes format for signing
        transaction_bytes = json.dumps(ordered_data_for_signing, sort_keys=True).encode('utf-8')

        # Use PSS padding which is recommended for signatures with RSA
        signature = self.private_key.sign(
            transaction_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        # Return the signature as a base64 encoded string
        return base64.b64encode(signature).decode('utf-8')

    def get_public_key_string(self) -> str:
        """Returns the public key in PEM format as a string."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo # Standard format
        ).decode('utf-8')

    def get_address(self) -> str:
        """Generates a simple address for the wallet based on the public key."""
        # Create a simple address by hashing the DER representation of the public key
        pub_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        # Take the first 40 characters of the SHA256 hash as the address (common practice for simplicity)
        return hashlib.sha256(pub_bytes).hexdigest()[:40]

def verify_signature(public_key_pem: str, data: Dict, signature: str) -> bool:
    """Verifies a signature against data using a public key."""
    try:
        # Load the public key from PEM format string into a public key object
        public_key = serialization.load_pem_public_key(public_key_pem.encode('utf-8'))

        # Prepare the data for verification (must be in the same ordered format as for signing)
        # Ensure the data is in bytes format for verification
        data_bytes = json.dumps(OrderedDict(sorted(data.items())), sort_keys=True).encode('utf-8')
        # Decode the base64 signature string back to bytes
        signature_bytes = base64.b64decode(signature)

        # Verify the signature using the public key
        public_key.verify(
            signature_bytes,
            data_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return True # Signature is valid
    except (InvalidSignature, Exception) as e:
        # Catch InvalidSignature specifically (signature does not match)
        # Also catch general exceptions during the process (e.g., invalid key format)
        logger.warning(f"Signature verification failed: {e}")
        return False # Signature is invalid or verification failed

# ============================
# Merkle Tree Implementation
# ============================
def build_merkle_tree(transactions: List[Dict]) -> Optional[str]:
    """
    Builds a Merkle Tree from a list of transaction dictionaries and returns the Merkle Root hash.
    Each leaf node is the hash of a transaction dictionary.
    Duplicates the last hash if the number of leaves is odd at any level.
    """
    if not transactions:
        return None # Return None for an empty list of transactions

    # Calculate the hash for each transaction dictionary to form the leaf nodes
    leaf_hashes = [hash_dict(tx) for tx in transactions]

    # If there are no transactions after hashing (shouldn't happen if input is not empty, but safety check)
    if not leaf_hashes:
         return None

    # Merkle tree construction loop: repeatedly hash pairs of nodes
    level = leaf_hashes
    while len(level) > 1:
        next_level = []
        # Process nodes in pairs
        for i in range(0, len(level), 2):
            left = level[i]
            # If there is no right node (odd number of nodes), duplicate the left node
            right = level[i+1] if i+1 < len(level) else left
            # Concatenate the hashes and calculate the hash of the combined string
            combined_hash = calculate_sha256(left + right)
            next_level.append(combined_hash)
        level = next_level # Move to the next level of the tree

    # The final remaining hash is the Merkle Root
    return level[0] if level else None # Return the root hash, or None if something went wrong

# ============================
# Block and Blockchain Classes
# ============================
class Transaction:
    """Represents a single transaction in the blockchain."""
    def __init__(self, sender: str, recipient: str, amount: float,
                 fee: float, public_key: str, signature: Optional[str] = None,
                 timestamp: Optional[float] = None, tx_hash: Optional[str] = None):
        self.sender = sender
        self.recipient = recipient
        # Ensure amount and fee are stored as floats
        self.amount = float(amount)
        self.fee = float(fee)
        # Use provided timestamp or current time if not provided
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.public_key = public_key
        self.signature = signature
        # Calculate hash upon initialization if not provided
        self.tx_hash = tx_hash if tx_hash is not None else self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculates the SHA256 hash of the transaction content."""
        # Ensure the hash is calculated from a consistent, ordered representation
        # The signature is NOT included in the hash calculation
        tx_dict = OrderedDict([
            ("sender", self.sender), ("recipient", self.recipient),
            ("amount", self.amount), ("fee", self.fee),
            ("timestamp", self.timestamp), ("public_key", self.public_key)
        ])
        return hash_dict(tx_dict)

    def sign(self, wallet: Wallet) -> bool:
        """Signs the transaction using the sender's wallet."""
        # Transaction must be signed by the sender's wallet address
        if wallet.get_address() != self.sender:
            logger.warning(f"Signing failed: Wallet address {wallet.get_address()[:10]}... does not match sender address {self.sender[:10]}...")
            return False

        # Prepare the data that will be signed (same as what's used for hashing, excluding signature)
        signable_data = OrderedDict([
            ("sender", self.sender), ("recipient", self.recipient),
            ("amount", self.amount), ("fee", self.fee),
            ("timestamp", self.timestamp), ("public_key", self.public_key)
        ])

        # Generate the signature using the wallet's private key
        self.signature = wallet.sign_transaction(signable_data)

        # Recalculate hash after signing to ensure tx_hash reflects the signed state
        # Although signature isn't in _calculate_hash input, this updates tx_hash in the object
        self.tx_hash = self._calculate_hash()

        return True

    def is_valid(self) -> bool:
        """Validates the transaction, including signature verification and hash consistency."""
        # Basic checks for required fields
        if not all([self.sender, self.recipient, self.public_key, self.signature is not None]):
            logger.debug(f"TX Invalid ({self.tx_hash[:10]}...): Missing fields. Sender: {bool(self.sender)}, Recipient: {bool(self.recipient)}, PubKey: {bool(self.public_key)}, Sig: {bool(self.signature)}")
            return False

        # Amount and fee validation: must be non-negative
        if self.amount < 0 or self.fee < 0:
            logger.debug(f"TX Invalid ({self.tx_hash[:10]}...): Negative amount ({self.amount}) or fee ({self.fee}).")
            return False

        # Verify the transaction hash matches its content: ensures data integrity of the core data
        # This check must happen regardless of whether it's a special transaction or not
        calculated_hash = self._calculate_hash()
        if self.tx_hash != calculated_hash:
             logger.debug(f"TX Invalid ({self.tx_hash[:10]}...): Hash mismatch. Stored: {self.tx_hash[:10]}, Calc: {calculated_hash[:10]}")
             return False


        # Special handling for genesis and coinbase transactions (sender is "0")
        if self.sender == "0":
            # Genesis and Coinbase transactions have specific, non-cryptographic signatures in this demo
            if self.signature in ["genesis_signature", "coinbase"]:
                 # Basic check: Ensure recipient is not empty for coinbase
                 if self.signature == "coinbase" and not self.recipient:
                      logger.debug(f"TX Invalid ({self.tx_hash[:10]}...): Coinbase transaction with empty recipient.")
                      return False
                 # Further validation for coinbase/genesis could include checking reward amount, etc.
                 return True # Valid special transaction
            else:
                logger.debug(f"TX Invalid ({self.tx_hash[:10]}...): Sender '0' but invalid special signature '{self.signature}'.")
                return False # Sender "0" must have a known special signature

        # --- Standard user transaction validation ---
        # 1. Verify the signature: ensures the transaction was signed by the sender's private key
        verifiable_data = OrderedDict([
            ("sender", self.sender), ("recipient", self.recipient),
            ("amount", self.amount), ("fee", self.fee),
            ("timestamp", self.timestamp), ("public_key", self.public_key)
        ])
        if not verify_signature(self.public_key, verifiable_data, self.signature):
            logger.debug(f"TX Invalid ({self.tx_hash[:10]}...): Signature verification failed.")
            return False

        # 3. (In a real system) Check if the sender has sufficient balance.
        # This requires querying the blockchain state, which is outside the scope of a single Transaction object's validation.
        # This check would typically happen before adding to the transaction pool or during block validation.

        return True # Transaction is valid

    def to_dict(self) -> Dict:
        """Returns a dictionary representation of the transaction suitable for serialization."""
        # Recalculate hash to ensure the dictionary representation has the correct hash value
        self.tx_hash = self._calculate_hash()
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp,
            "public_key": self.public_key,
            "signature": self.signature,
            "tx_hash": self.tx_hash # Include hash for convenience and lookup
        }

    @classmethod
    def from_dict(cls, tx_dict: Dict) -> 'Transaction':
        """
        Creates a Transaction object from a dictionary.
        Raises ValueError if required keys are missing or data is invalid.
        """
        # Define required keys
        required_keys = ['sender', 'recipient', 'amount', 'fee', 'public_key']
        # Define optional keys with default values (None or 0.0)
        optional_keys = {'signature': None, 'timestamp': None, 'tx_hash': None}

        # Check for missing required keys and validate types
        for key in required_keys:
            if key not in tx_dict:
                # Log the error before raising
                logger.error(f"Transaction dictionary missing required key: '{key}'. Dict: {tx_dict}")
                raise ValueError(f"Missing required key '{key}' in transaction dictionary.")
            # Basic type checks for critical fields
            if key in ['amount', 'fee']:
                 if not isinstance(tx_dict[key], (int, float)):
                      logger.error(f"Transaction dictionary has invalid type for '{key}'. Expected int or float. Dict: {tx_dict}")
                      raise ValueError(f"Invalid type for '{key}' in transaction dictionary. Expected int or float.")
            elif key == 'public_key':
                 if not isinstance(tx_dict[key], str):
                      logger.error(f"Transaction dictionary has invalid type for '{key}'. Expected string. Dict: {tx_dict}")
                      raise ValueError(f"Invalid type for '{key}' in transaction dictionary. Expected string.")


        # Create the Transaction object using dictionary unpacking for known keys
        # Use .get() with default for optional keys to avoid KeyError if they are missing
        try:
            # Ensure numeric values are floats
            amount = float(tx_dict['amount'])
            fee = float(tx_dict['fee'])
            timestamp = float(tx_dict['timestamp']) if tx_dict.get('timestamp') is not None else None # Handle optional timestamp

            tx = cls(
                sender=tx_dict['sender'],
                recipient=tx_dict['recipient'],
                amount=amount,
                fee=fee,
                public_key=tx_dict['public_key'],
                signature=tx_dict.get('signature', optional_keys['signature']),
                timestamp=timestamp,
                tx_hash=tx_dict.get('tx_hash', optional_keys['tx_hash']) # Load stored hash if available
            )

            # If tx_hash was not provided (or was None), calculate it
            if tx.tx_hash is None:
                tx.tx_hash = tx._calculate_hash()
            else:
                 # If tx_hash was provided, validate that it matches the calculated hash
                 calculated_hash = tx._calculate_hash()
                 if tx.tx_hash != calculated_hash:
                      # Log a warning but don't raise error here. Validation will catch this later.
                      logger.warning(f"Transaction hash mismatch during loading! Stored: {tx.tx_hash[:10]}, Calculated: {calculated_hash[:10]}.")
                      # Optionally, update the object's hash to the calculated one for internal consistency:
                      # tx.tx_hash = calculated_hash # Decide if you want to correct or just flag


            # Note: is_valid() is NOT called automatically here.
            # Validation (including signature and hash check) should be done after creating the object,
            # typically when adding to the pool or validating a block.

            return tx
        except ValueError as e:
            # Catch any ValueErrors from float conversion or other checks
            logger.error(f"Value error creating Transaction from dict: {tx_dict}. Error: {e}")
            raise ValueError(f"Failed to create Transaction object from dictionary due to value error: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during object creation
            logger.error(f"Unexpected error creating Transaction from dict: {tx_dict}. Error: {e}")
            raise ValueError(f"Failed to create Transaction object from dictionary: {e}") from e


class Block:
    """Represents a single block in the blockchain."""
    def __init__(self, index: int, transactions: List[Dict], previous_hash: str,
                 difficulty: int, timestamp: Optional[float] = None, nonce: int = 0,
                 block_hash: Optional[str] = None, merkle_root: Optional[str] = None): # Include merkle_root as optional param for loading
        self.index = index
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.transactions = transactions # Stored as list of transaction dictionaries
        self.previous_hash = previous_hash
        self.difficulty = difficulty
        self.nonce = nonce
        # Merkle root is calculated from the list of transaction dictionaries.
        # Calculate it upon initialization unless provided (e.g., when loading from DB).
        self.merkle_root = merkle_root if merkle_root is not None else build_merkle_tree(self.transactions)
        # Calculate block hash upon initialization unless provided (e.g., when loading from DB)
        self.block_hash = block_hash if block_hash is not None else self.calculate_block_hash()

    def calculate_block_hash(self) -> str:
        """Calculates the SHA256 hash of the block's header content."""
        # Ensure the hash is calculated from a consistent, ordered representation
        block_content = OrderedDict([
            ("index", self.index),
            ("timestamp", self.timestamp),
            ("merkle_root", self.merkle_root), # Include the Merkle Root
            ("previous_hash", self.previous_hash),
            ("nonce", self.nonce),
            ("difficulty", self.difficulty)
            # The full transactions list is NOT included directly in the block hash calculation, only its merkle_root
            # The block_hash itself is NOT included in the hash calculation input
        ])
        return hash_dict(block_content)

    def mine_block(self, difficulty: int) -> Tuple[str, float]: # Returns final hash and mining_time
        """Mines the block to find a hash that meets the required difficulty."""
        self.difficulty = difficulty # Set the difficulty for this mining attempt
        # The target hash prefix is a string of '0's of length equal to the difficulty
        prefix = "0" * self.difficulty
        logger.info(f"Mining block {self.index} with difficulty {self.difficulty} (prefix: '{prefix}')...")
        start_time = time.time() # Record start time for mining duration calculation
        self.nonce = 0 # Reset nonce for each mining attempt

        # Add a reasonable search limit for demo purposes to prevent infinite loops on high difficulty
        # This limit is arbitrary and depends on the target difficulty and computational power.
        effective_search_limit = 10_000_000 # A larger limit than before, still finite
        # For very low difficulties (1 or 2), the hash might be found quickly, no high limit needed.
        # For difficulty 3 or higher, the search space grows exponentially.
        # If difficulty is high, a very large limit is needed or consider alternative PoW like in real blockchains.
        # For this demo, let's make the limit difficulty-aware but capped
        if self.difficulty >= 4: # For difficulties 4 and above, use a higher cap but warn
             effective_search_limit = 50_000_000 # Increased cap for higher difficulty simulation
             logger.warning(f"Mining difficulty {self.difficulty} is high, using a nonce search limit of {effective_search_limit}. Mining might take a long time or fail.")
        elif self.difficulty <= 2:
             effective_search_limit = 1_000_000 # Lower limit for easier difficulties

        logger.debug(f"Mining Block {self.index}: Starting nonce search up to {effective_search_limit}")


        while True:
            # Calculate the hash of the block with the current nonce
            current_hash = self.calculate_block_hash()

            # Check if the calculated hash meets the difficulty requirement
            if current_hash.startswith(prefix):
                break # Mining successful: the current_hash is the valid block hash

            self.nonce += 1 # Increment nonce to try a different hash

            # Check against the search limit to prevent excessive computation time
            if self.nonce >= effective_search_limit:
                mining_time = time.time() - start_time
                logger.error(f"Mining failed for block {self.index} after {mining_time:.2f}s. Nonce {self.nonce} reached search limit {effective_search_limit}. Difficulty: {self.difficulty}")
                # It's important to indicate failure if the limit is reached
                raise Exception(f"Mining failed for block {self.index}: Nonce limit reached.")

        self.block_hash = current_hash # Store the final valid hash in the block object
        mining_time = time.time() - start_time # Calculate the total mining time
        logger.info(f"Block {self.index} mined successfully in {mining_time:.2f} seconds. Nonce: {self.nonce}, Hash: {self.block_hash[:12]}...")
        return self.block_hash, mining_time

    def to_dict(self) -> Dict:
        """Returns a dictionary representation of the block suitable for serialization."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions, # Store list of transaction dictionaries
            "previous_hash": self.previous_hash,
            "difficulty": self.difficulty,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "block_hash": self.block_hash # Include calculated hash for storage
        }

    @classmethod
    def from_dict(cls, block_dict: Dict) -> 'Block':
        """
        Creates a Block object from a dictionary.
        This method loads data from a source (like a database), but does NOT perform
        full validation (hash links, PoW, Merkle root consistency). Validation
        should be done separately using `is_chain_valid`.
        Raises ValueError if required keys are missing.
        """
        # Define required keys for a block dictionary
        required_keys = ['index', 'transactions', 'previous_hash', 'difficulty', 'nonce']
        # Define optional keys (that might be calculated or added later)
        optional_keys = {'timestamp': None, 'block_hash': None, 'merkle_root': None}

        # Check for missing required keys
        for key in required_keys:
            if key not in block_dict:
                 logger.error(f"Block dictionary missing required key: '{key}'. Dict: {block_dict}")
                 raise ValueError(f"Missing required key '{key}' in block dictionary.")

        # Create the Block object using dictionary unpacking and .get() for optional keys
        try:
            block = cls(
                index=block_dict['index'],
                # transactions should already be a list of dictionaries when loaded
                transactions=block_dict['transactions'],
                previous_hash=block_dict['previous_hash'],
                difficulty=block_dict['difficulty'],
                nonce=block_dict['nonce'],
                # Use .get() with a default of None for optional keys
                timestamp=block_dict.get('timestamp', optional_keys['timestamp']),
                block_hash=block_dict.get('block_hash', optional_keys['block_hash']), # Load stored hash
                merkle_root=block_dict.get('merkle_root', optional_keys['merkle_root']) # Load stored merkle root
            )

            # Note: We do NOT recalculate and overwrite block_hash or merkle_root here.
            # The loaded block object will contain the stored values.
            # Validation happens in `is_chain_valid` by comparing stored vs. calculated.

            # Basic type validation for critical fields
            if not isinstance(block.index, int) or not isinstance(block.difficulty, int) or not isinstance(block.nonce, int):
                 logger.error(f"Block dictionary has invalid type for index, difficulty, or nonce. Dict: {block_dict}")
                 raise ValueError(f"Invalid type for index, difficulty, or nonce in block dictionary.")
            if not isinstance(block.transactions, list):
                 logger.error(f"Block dictionary 'transactions' is not a list. Dict: {block_dict}")
                 raise ValueError(f"Invalid type for 'transactions' in block dictionary (expected list).")

            return block
        except Exception as e:
            # Catch any other unexpected errors during object creation
            logger.error(f"Unexpected error creating Block from dict: {block_dict}. Error: {e}")
            raise ValueError(f"Failed to create Block object from dictionary: {e}") from e


class TransactionPool:
    """Manages pending transactions before they are included in a block."""
    def __init__(self):
        # List to hold pending Transaction objects
        self.pending_transactions: List[Transaction] = []
        # Lock for thread-safe access to the pool (important in concurrent environments)
        self._lock = threading.Lock()

    def add_transaction(self, transaction: Transaction) -> bool:
        """Adds a valid and unique transaction object to the pool."""
        # First, validate the transaction object itself
        if not transaction.is_valid():
            logger.warning(f"Rejected invalid transaction: {transaction.tx_hash[:10]}...")
            return False

        with self._lock: # Acquire lock for thread-safe access
            # Check if a transaction with the same hash already exists in the pool
            if any(tx.tx_hash == transaction.tx_hash for tx in self.pending_transactions):
                logger.warning(f"Rejected duplicate transaction: {transaction.tx_hash[:10]}...")
                return False

            # Check if the transaction is already mined (optional, but good practice if integrating with a chain)
            # This would require access to the blockchain to check if the hash exists in any block.
            # For this demo, we assume transactions in the pool are not yet mined.

            self.pending_transactions.append(transaction) # Add the transaction object to the list
            logger.info(f"Transaction added to pool: {transaction.tx_hash[:10]}..., Fee: {transaction.fee}")
            return True

    def get_transactions(self, count: int) -> List[Dict]:
        """
        Returns a list of transaction dictionaries from the pool, sorted by fee (descending)
        then timestamp (ascending), up to the specified count.
        Transactions remain in the pool after being returned by this method.
        """
        with self._lock: # Acquire lock for thread-safe access
            # Sort transactions: highest fee first, then oldest timestamp first for ties (standard mempool behavior)
            # Using a tuple for the key allows sorting by multiple criteria
            sorted_txs = sorted(self.pending_transactions, key=lambda tx: (-tx.fee, tx.timestamp))

            # Return a list of transaction dictionaries for inclusion in a block
            # Return up to 'count' transactions
            return [tx.to_dict() for tx in sorted_txs[:count]]

    def remove_transactions(self, tx_hashes: List[str]):
        """Removes transactions matching the given list of hashes from the pool."""
        with self._lock: # Acquire lock for thread-safe access
            initial_count = len(self.pending_transactions)
            # Filter out transactions whose hashes are in the list to be removed
            self.pending_transactions = [tx for tx in self.pending_transactions if tx.tx_hash not in tx_hashes]
            removed_count = initial_count - len(self.pending_transactions)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} mined transaction(s) from the pool.")

    def get_pending_count(self) -> int:
        """Returns the current number of transactions in the pending pool."""
        with self._lock:
            return len(self.pending_transactions)


class BlockchainDB:
    """Handles database interactions for storing blockchain data (blocks and transactions)."""
    def __init__(self, db_path: str):
        # For this demo, we enforce using an in-memory database for simplicity and non-persistence
        # In a real application, you would use the provided db_path for file storage like:
        # self.db_path_str = db_path
        self.db_path_str = ":memory:"
        # Log that we are using in-memory and ignoring the provided path
        logger.info(f"Using in-memory SQLite database for the demo. Original path '{db_path}' is ignored for persistence.")

        # Connect to the SQLite database. `check_same_thread=False` is needed for multithreaded access,
        # which might occur if mining runs in a separate thread (though not explicitly done in this main demo flow).
        self._conn = sqlite3.connect(self.db_path_str, check_same_thread=False)
        # Configure row factory to return rows as dictionaries (sqlite3.Row) for easier access by column name
        self._conn.row_factory = sqlite3.Row

        # Initialize the database schema (create tables if they don't exist)
        self._init_db()

    def _init_db(self):
        """Creates the necessary tables (blocks and transactions) in the database."""
        cursor = self._conn.cursor()
        # Table for blocks
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS blocks (
            block_hash TEXT PRIMARY KEY,
            [index] INTEGER UNIQUE NOT NULL, -- Index of the block, unique for chain order
            timestamp REAL NOT NULL,       -- Unix timestamp of block creation/mining
            previous_hash TEXT NOT NULL,   -- Hash of the previous block
            merkle_root TEXT,              -- Merkle root of transactions in the block
            nonce INTEGER NOT NULL,        -- Nonce found during mining
            difficulty INTEGER NOT NULL,   -- Difficulty level for this block
            transactions_data TEXT NOT NULL -- JSON string representation of the list of transaction dictionaries
        )''')

        # Table for transactions (to quickly query transactions by address and link to blocks)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            tx_hash TEXT PRIMARY KEY,       -- Unique hash of the transaction
            block_hash TEXT,                -- Hash of the block this transaction is in (NULL for pending txs - not used in this DB impl for pool)
            sender TEXT NOT NULL,           -- Sender address
            recipient TEXT NOT NULL,        -- Recipient address
            amount REAL NOT NULL,           -- Amount transferred
            fee REAL NOT NULL,              -- Transaction fee
            timestamp REAL NOT NULL,        -- Unix timestamp of transaction creation
            public_key TEXT NOT NULL,       -- Sender's public key (in PEM format)
            signature TEXT NOT NULL,        -- Transaction signature (base64 encoded)
            full_tx_data TEXT NOT NULL,     -- Stores the full JSON string representation of the transaction dictionary
            FOREIGN KEY (block_hash) REFERENCES blocks(block_hash) -- Link to the blocks table
        )''')

        # Commit the schema changes
        self._conn.commit()
        logger.debug("Database schema initialized (in-memory).")

    def save_block(self, block: Block):
        """Saves a Block object and its associated transactions to the database."""
        cursor = self._conn.cursor()
        try:
            # Serialize the block's transaction list into a JSON string for storage
            transactions_json = json.dumps(block.transactions)

            # Insert or replace the block data into the 'blocks' table
            # Using INSERT OR REPLACE simplifies saving if the block already exists (e.g., re-saving during tampering demo)
            cursor.execute(
                'INSERT OR REPLACE INTO blocks (block_hash, [index], timestamp, previous_hash, merkle_root, nonce, difficulty, transactions_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (block.block_hash, block.index, block.timestamp, block.previous_hash,
                 block.merkle_root, block.nonce, block.difficulty, transactions_json)
            )

            # Insert or replace each transaction associated with this block into the 'transactions' table
            for tx_dict in block.transactions:
                # Ensure tx_hash is present in the transaction dictionary (it should be, based on Transaction.to_dict)
                tx_hash_to_save = tx_dict.get("tx_hash")
                if not tx_hash_to_save:
                    # Log an error if a transaction dictionary is missing its hash (indicates a problem earlier)
                    logger.error(f"Transaction dictionary missing 'tx_hash' for block {block.index}. Skipping save for this transaction: {tx_dict}")
                    continue # Skip saving this specific transaction

                # Store the full transaction dictionary as a JSON string
                full_tx_data_json = json.dumps(tx_dict)

                # Insert or replace the transaction data
                cursor.execute(
                    'INSERT OR REPLACE INTO transactions (tx_hash, block_hash, sender, recipient, amount, fee, timestamp, public_key, signature, full_tx_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (tx_hash_to_save, block.block_hash, tx_dict.get("sender"), tx_dict.get("recipient"),
                     tx_dict.get("amount"), tx_dict.get("fee"), tx_dict.get("timestamp"),
                     tx_dict.get("public_key"), tx_dict.get("signature"), full_tx_data_json)
                )

            self._conn.commit() # Commit all changes (block and transactions)
            logger.info(f"Block {block.index} ({block.block_hash[:10]}...) saved successfully to DB.")
        except Exception as e:
            # Log and rollback changes if any part of the saving process fails
            logger.error(f"Error saving block {block.index} to DB: {e}")
            self._conn.rollback() # Discard partial changes

    def _db_row_to_block(self, row: sqlite3.Row) -> Optional[Block]:
        """Helper function to convert a database row from the 'blocks' table into a Block object."""
        if not row:
            return None # Return None if the input row is empty or None
        try:
            # Convert the sqlite3.Row object (which behaves like a dict) to a standard dictionary
            block_dict = dict(row)
            # Parse the JSON string back into a list of transaction dictionaries
            block_dict["transactions"] = json.loads(block_dict["transactions_data"])
            # Remove the raw JSON string key as it's not part of the Block object's constructor
            del block_dict["transactions_data"]

            # Create a Block object from the dictionary using the from_dict class method
            # The from_dict method will load the stored block_hash and merkle_root
            return Block.from_dict(block_dict)
        except Exception as e:
            # Log an error if conversion fails and return None
            logger.error(f"Error converting DB row to Block object: {e}. Row data (first 100 chars): {str(dict(row))[:100]}...")
            return None # Indicate failure

    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Retrieves a block from the database by its hash."""
        cursor = self._conn.cursor()
        cursor.execute('SELECT * FROM blocks WHERE block_hash = ?', (block_hash,))
        # Fetch one row and convert it to a Block object using the helper function
        return self._db_row_to_block(cursor.fetchone())

    def get_latest_block(self) -> Optional[Block]:
        """Retrieves the block with the highest index (the tip of the chain) from the database."""
        cursor = self._conn.cursor()
        # Order by index in descending order and limit to 1 to get the highest index block
        cursor.execute('SELECT * FROM blocks ORDER BY [index] DESC LIMIT 1')
        # Fetch one row and convert it to a Block object
        return self._db_row_to_block(cursor.fetchone())

    def get_blocks_range(self, start_index: int, end_index: int) -> List[Block]:
        """Retrieves blocks from the database within a specific index range (inclusive)."""
        cursor = self._conn.cursor()
        # Select blocks where the index is within the specified range, ordered by index ascending
        cursor.execute('SELECT * FROM blocks WHERE [index] >= ? AND [index] <= ? ORDER BY [index] ASC', (start_index, end_index))
        # Fetch all matching rows, then use a list comprehension and map to convert each row to a Block object
        # Filter out any potential None results from _db_row_to_block
        return [b for b in map(self._db_row_to_block, cursor.fetchall()) if b is not None]

    def get_block_by_index_db(self, index_val: int) -> Optional[Block]:
        """Retrieves a block from the database by its index."""
        cursor = self._conn.cursor()
        cursor.execute('SELECT * FROM blocks WHERE [index] = ?', (index_val,))
        # Fetch one row and convert it to a Block object
        return self._db_row_to_block(cursor.fetchone())

    def get_address_transactions(self, address: str) -> List[Dict]:
        """Retrieves all transaction dictionaries related to a specific address (as sender or recipient) from the database."""
        cursor = self._conn.cursor()
        # Select the full JSON transaction data for transactions where the address is either sender or recipient
        cursor.execute('SELECT full_tx_data FROM transactions WHERE sender = ? OR recipient = ? ORDER BY timestamp ASC', (address, address))
        # Fetch all rows and parse the JSON string (`full_tx_data`) from each row into a Python dictionary
        return [json.loads(row['full_tx_data']) for row in cursor.fetchall()]

    def close_db(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None # Set connection to None to indicate it's closed
            logger.info("Database connection closed.")

class Blockchain:
    """Core Blockchain class managing blocks, transactions, mining, and validation."""
    def __init__(self, db_path: Optional[str] = None):
        # Load difficulty and other parameters from the config file or use fallbacks
        # Convert config values to appropriate types (int, float)
        self.difficulty = int(config.get("BLOCKCHAIN", "difficulty", fallback="3"))
        self.max_transactions_per_block = int(config.get("BLOCKCHAIN", "max_transactions_per_block", fallback="5"))
        self.block_time_target = float(config.get("BLOCKCHAIN", "block_time_target_seconds", fallback="15"))
        self.difficulty_adjustment_interval = int(config.get("BLOCKCHAIN", "difficulty_adjustment_interval", fallback="5"))
        self.miner_reward = float(config.get("BLOCKCHAIN", "miner_reward", fallback="50.0")) # Add a config option for base reward

        # Initialize the database - db_path is effectively ignored by BlockchainDB for the demo (uses :memory:)
        # Pass the config path as the theoretical db_path for the DB class
        db_storage_path = db_path or config.get("DATABASE", "path", fallback="bc.db")
        self.db = BlockchainDB(db_storage_path)

        # Initialize the transaction pool
        self.transaction_pool = TransactionPool()

        # List to store block creation details for tabular output in the demo
        self.block_creation_log: List[Dict[str, Any]] = []

        # Create the genesis block if the chain is empty (no blocks in the DB)
        if not self.db.get_latest_block():
            self._create_genesis_block()

    def _create_genesis_block(self):
        """Creates and mines the first block (genesis block) of the blockchain."""
        logger.info("Creating genesis block...")

        # Define the genesis transaction dictionary
        # Sender "0" signifies a special system transaction (like coinbase)
        # "genesis_signature" is a placeholder, not a cryptographic signature
        genesis_tx_dict = {
            "sender": "0", # Special sender address for system transactions
            "recipient": "genesis_address", # A designated address for initial coins (if any) or simply a placeholder
            "amount": self.miner_reward, # Example: Initial reward goes to genesis address (or could be 0)
            "fee": 0.0,
            "public_key": "", # No public key for sender "0"
            "signature": "genesis_signature", # A special, non-cryptographic signature for the genesis transaction
            "timestamp": time.time() # Use current time for the genesis block transaction
        }
        # Create a Transaction object from the dictionary
        try:
             genesis_tx = Transaction.from_dict(genesis_tx_dict)
        except ValueError as e:
             logger.critical(f"FATAL ERROR: Failed to create genesis transaction from dict: {e}"); raise # Should not happen if dict is defined correctly

        # Create the genesis block object
        # Index is 0, previous_hash is "0" as there is no preceding block
        genesis_block = Block(0, [genesis_tx.to_dict()], "0", self.difficulty)

        try:
            # Mine the genesis block using the initial difficulty
            # The mining process finds the nonce and calculates the block hash
            _, mining_time = genesis_block.mine_block(genesis_block.difficulty)

            # Save the successfully mined genesis block to the database
            self.db.save_block(genesis_block)

            # Add details about the genesis block creation to the log for the demo output table
            self.block_creation_log.append({
                "Index": genesis_block.index,
                "Timestamp": f"{genesis_block.timestamp:.0f}",
                "Mined By": "System", # Indicate it's the system-created genesis block
                "Nonce": genesis_block.nonce,
                "Difficulty": genesis_block.difficulty,
                "# Txs": len(genesis_block.transactions),
                "Block Hash": genesis_block.block_hash[:12]+"...", # Truncate hash for display
                "Mining Time (s)": f"{mining_time:.2f}",
                "Notes": "Genesis Block"
            })
            logger.info(f"Genesis block created and saved. Hash: {genesis_block.block_hash[:12]}...")
        except Exception as e:
            # Log a critical error and re-raise if genesis block creation or saving fails
            logger.critical(f"FATAL ERROR: Failed to mine/save genesis block: {e}"); raise

    def add_transaction(self, transaction: Transaction) -> bool:
        """Adds a transaction object to the pending transaction pool."""
        # The TransactionPool's add_transaction method already includes validation checks
        return self.transaction_pool.add_transaction(transaction)

    def mine_pending_transactions(self, miner_address: str) -> Optional[Block]:
        """
        Mines a new block containing pending transactions from the pool.
        Includes a coinbase transaction for the miner reward and collected fees.
        Returns the newly mined block object or None if mining fails or no transactions are available (besides coinbase).
        """
        latest_block = self.db.get_latest_block()
        if not latest_block:
            logger.error("Cannot mine: Blockchain is empty or genesis block not found.")
            return None

        current_difficulty = self.difficulty # Start with the current blockchain difficulty
        # Check if it's time for a difficulty adjustment based on the index of the *next* block
        # Difficulty is adjusted before mining the block that completes the interval
        if (latest_block.index + 1) > 0 and (latest_block.index + 1) % self.difficulty_adjustment_interval == 0:
             logger.info(f"Block {latest_block.index + 1}: Checking for difficulty adjustment...")
             # Calculate the new difficulty for the *next* block to be mined
             current_difficulty = self._adjust_difficulty(latest_block)
             self.difficulty = current_difficulty # Update the blockchain's current difficulty property

        # Get pending transactions from the pool. Reserve 1 slot for the coinbase transaction.
        # `get_transactions` returns transaction dictionaries
        pending_tx_dicts = self.transaction_pool.get_transactions(self.max_transactions_per_block - 1)

        # Calculate the total fees from the selected pending user transactions
        total_fees = sum(tx_d.get("fee", 0.0) for tx_d in pending_tx_dicts)

        # Create the coinbase transaction dictionary (miner reward + collected fees)
        # Sender "0" and signature "coinbase" identify this as a special transaction
        coinbase_tx_dict = {
             "sender": "0",
             "recipient": miner_address,
             "amount": self.miner_reward + total_fees, # Base reward + fees from user transactions
             "fee": 0.0, # Coinbase transaction itself has no fee
             "public_key": "", # Sender "0" has no public key
             "signature": "coinbase", # Special signature for coinbase transaction
             "timestamp": time.time() # Timestamp for the coinbase transaction
        }
        # Create a Transaction object for the coinbase transaction
        try:
             coinbase_tx_obj = Transaction.from_dict(coinbase_tx_dict)
        except ValueError as e:
             logger.error(f"Failed to create coinbase transaction from dict: {e}. Mining aborted.")
             return None # Abort mining if coinbase transaction creation fails

        # Convert the coinbase transaction object back to dictionary format for inclusion in the block
        coinbase_tx_dict_final = coinbase_tx_obj.to_dict()

        # Combine coinbase transaction with the selected pending transactions for the new block
        all_txs_for_block = [coinbase_tx_dict_final] + pending_tx_dicts

        # Check if there are any transactions (at least the coinbase) to include in the block
        if not all_txs_for_block:
             # This case should theoretically not be reached if coinbase is always added, but as a safeguard
             logger.warning("No transactions (including coinbase) prepared for the block. Skipping mining.")
             return None

        # Create the new block object
        # Index is one greater than the latest block, previous_hash is the hash of the latest block
        new_block = Block(latest_block.index + 1, all_txs_for_block, latest_block.block_hash, current_difficulty)

        try:
            # Mine the new block using the determined difficulty
            # This finds the valid nonce and calculates the block hash
            block_hash, mining_time = new_block.mine_block(current_difficulty)

            # Save the successfully mined block and its transactions to the database
            self.db.save_block(new_block)

            # Add block creation details to the log for the demo output table
            self.block_creation_log.append({
                "Index": new_block.index,
                "Timestamp": f"{new_block.timestamp:.0f}",
                "Mined By": miner_address[:8]+"...", # Truncate miner address for display
                "Nonce": new_block.nonce,
                "Difficulty": new_block.difficulty,
                "# Txs": len(new_block.transactions),
                "Block Hash": block_hash[:12]+"...", # Truncate block hash for display
                "Mining Time (s)": f"{mining_time:.2f}",
                "Notes": f"Reward ({self.miner_reward + total_fees:.2f}) + {len(pending_tx_dicts)} user txs"
            })

            # Remove the transactions that were successfully included in the block from the pending pool
            # Only remove the user transactions, not the coinbase transaction
            mined_tx_hashes = [d.get('tx_hash') for d in pending_tx_dicts if d.get('tx_hash')] # Get hashes of user transactions, filter None
            self.transaction_pool.remove_transactions(mined_tx_hashes)

            logger.info(f"Block {new_block.index} successfully mined by {miner_address[:8]}..., hash: {block_hash[:12]}... Difficulty: {new_block.difficulty}. Included {len(all_txs_for_block)} transactions.")
            return new_block # Return the newly mined block object

        except Exception as e:
            # Log an error if mining or saving the block fails
            logger.error(f"Failed to mine/save block {new_block.index}: {e}. Transactions remain in pool.")
            # Transactions are NOT removed from the pool if mining fails, allowing them to be included in the next mining attempt
            return None # Indicate that mining failed

    def _adjust_difficulty(self, latest_block: Block) -> int:
        """
        Adjusts the mining difficulty based on the average time taken to mine
        the last 'difficulty_adjustment_interval' blocks compared to the target block time.
        """
        # We need at least 'difficulty_adjustment_interval' blocks to perform the first adjustment
        if latest_block.index < self.difficulty_adjustment_interval - 1:
            logger.debug(f"Not enough blocks for difficulty adjustment (need {self.difficulty_adjustment_interval}, have {latest_block.index + 1}). Using current difficulty {self.difficulty}.")
            return self.difficulty # Return current difficulty if not enough blocks have been mined

        # Get the block that marks the beginning of the current adjustment interval
        first_block_idx = latest_block.index - self.difficulty_adjustment_interval + 1
        first_block = self.db.get_block_by_index_db(first_block_idx)

        # This is a safeguard; the block should exist in a valid chain history
        if not first_block:
            logger.error(f"Block {first_block_idx} not found for difficulty adjustment calculation. Difficulty unchanged ({self.difficulty}).");
            return self.difficulty

        # Calculate the actual time taken to mine the last 'difficulty_adjustment_interval' blocks
        # This is the difference between the timestamp of the latest block and the first block of the interval
        actual_time = latest_block.timestamp - first_block.timestamp
        # Calculate the expected total time for this interval based on the target block time
        expected_time = self.difficulty_adjustment_interval * self.block_time_target

        new_diff = self.difficulty # Start with the current difficulty as the candidate for adjustment

        logger.debug(f"Difficulty adjustment check (Blocks {first_block.index}-{latest_block.index}): Actual time: {actual_time:.2f}s, Expected time: {expected_time:.2f}s")

        # Avoid division by zero or negative actual time (shouldn't happen with valid timestamps, but safeguard)
        if actual_time <= 0:
            logger.warning("Time interval is zero or negative for difficulty adjustment. Difficulty unchanged.")
            return new_diff

        # Calculate the ratio of expected time to actual time.
        # If time_ratio > 1, blocks were mined faster than target, so increase difficulty.
        # If time_ratio < 1, blocks were mined slower than target, so decrease difficulty.
        time_ratio = expected_time / actual_time

        # Adjust difficulty: increase if faster, decrease if slower
        # The adjustment factor (e.g., 1.5 for increasing, 0.75 for decreasing) can be tuned
        if time_ratio > 1.1: # Example: If blocks are 10%+ faster than target
            new_diff += 1
            logger.info(f"Difficulty increasing to {new_diff} for the NEXT block. Avg block time: {actual_time/self.difficulty_adjustment_interval:.2f}s (faster than target)")
        elif time_ratio < 0.9: # Example: If blocks are 10%+ slower than target
            # Ensure difficulty does not go below 1
            new_diff = max(1, self.difficulty - 1)
            logger.info(f"Difficulty decreasing to {new_diff} for the NEXT block. Avg block time: {actual_time/self.difficulty_adjustment_interval:.2f}s (slower than target)")
        else:
            # If average time is within an acceptable range (e.g., +/- 10% of target), difficulty remains unchanged
            logger.info(f"Difficulty remains {new_diff} for the NEXT block. Avg block time: {actual_time/self.difficulty_adjustment_interval:.2f}s (within target range)")

        # In a real system, the updated difficulty would need to be agreed upon by the network
        # and consistently applied to the next block. In this simulation, we just update the instance variable.

        return new_diff # Return the calculated difficulty for the next block

    def is_chain_valid(self) -> Tuple[bool, Optional[str]]:
        """
        Validates the integrity of the entire blockchain from genesis to the latest block.
        Checks:
        1. Hash link: previous_hash of current block matches hash of previous block.
        2. Block hash: current block's stored hash matches its calculated hash (validates content integrity).
        3. Proof of Work (PoW): current block's stored hash satisfies the required difficulty.
        4. Merkle Root: current block's stored Merkle root matches the calculated root from its transactions.
        5. Transaction validity: all transactions within each block (excluding special cases like coinbase/genesis) are valid.

        Returns True and None if the chain is valid, False and an error message if invalid.
        """
        latest_db_block = self.db.get_latest_block()
        if not latest_db_block:
            logger.warning("Chain validation: No blocks found in DB. An empty chain is considered valid.")
            return True, None # An empty or only-genesis chain can be considered valid initially

        # Validate genesis block (index 0) first as a base case
        genesis_block = self.db.get_block_by_index_db(0)
        if not genesis_block:
            # Critical error: genesis block is missing
            return False, "Genesis block (index 0) not found in DB."

        logger.info("Validating blockchain integrity...")

        # Validate genesis block's internal consistency and PoW
        calculated_genesis_hash = genesis_block.calculate_block_hash()
        if genesis_block.block_hash != calculated_genesis_hash:
            return False, f"Genesis block hash mismatch. Stored: {genesis_block.block_hash[:8]}, Calc: {calculated_genesis_hash[:8]}"
        if not genesis_block.block_hash.startswith("0" * genesis_block.difficulty):
             return False, f"Genesis block Proof of Work invalid. Hash: {genesis_block.block_hash[:8]}, Required Diff: {genesis_block.difficulty}"
        calculated_genesis_merkle_root = build_merkle_tree(genesis_block.transactions)
        if genesis_block.merkle_root != calculated_genesis_merkle_root:
             return False, f"Genesis block Merkle root mismatch. Stored: {genesis_block.merkle_root[:8] if genesis_block.merkle_root else 'None'}, Calc: {calculated_genesis_merkle_root[:8] if calculated_genesis_merkle_root else 'None'}"

        # Validate genesis transaction(s) - assuming the first transaction is the special genesis tx
        # Note: The first transaction in the genesis block is typically the genesis transaction itself.
        if not genesis_block.transactions:
             return False, "Genesis block contains no transactions."
        genesis_tx_in_block = Transaction.from_dict(genesis_block.transactions[0])
        # Perform basic checks on the genesis transaction
        if genesis_tx_in_block.sender != "0" or genesis_tx_in_block.signature != "genesis_signature":
             return False, f"Genesis block's first transaction is not a valid genesis transaction."
        # You might add checks for the recipient and initial amount here as per genesis rules

        # Iterate and validate blocks from index 1 up to the latest block
        # Fetch blocks one by one or in small batches to manage memory for large chains
        for i in range(1, latest_db_block.index + 1):
            block = self.db.get_block_by_index_db(i)
            prev_block = self.db.get_block_by_index_db(i - 1)

            # Ensure both the current and previous blocks were successfully retrieved
            if not block:
                return False, f"Block {i} not found in DB during validation."
            if not prev_block:
                 # This indicates a severe database issue or corrupted chain state
                 return False, f"Previous block {i-1} not found in DB for block {i} during validation."

            # --- Perform Validation Checks ---

            # 1. Check the hash link: previous_hash of the current block must match the block_hash of the previous block
            if block.previous_hash != prev_block.block_hash:
                return False, f"Block {i} previous_hash mismatch. Stored: {block.previous_hash[:8]}, Expected (from block {i-1}): {prev_block.block_hash[:8]}"

            # 2. Check the current block's own hash: Stored block_hash must match the hash calculated from its current content
            calculated_block_hash = block.calculate_block_hash()
            if block.block_hash != calculated_block_hash:
                # This means the block's content (index, timestamp, merkle_root, previous_hash, nonce, difficulty) has changed
                return False, f"Block {i} hash mismatch. Stored: {block.block_hash[:8]}, Calculated (from content): {calculated_block_hash[:8]}"

            # 3. Check Proof of Work (PoW): The block's stored hash must satisfy the difficulty requirement stored in the block
            # Use the stored difficulty from the block being validated
            difficulty_prefix = "0" * block.difficulty
            if not block.block_hash.startswith(difficulty_prefix):
                 # This indicates the block was not correctly mined or the hash was altered
                 return False, f"Block {i} Proof of Work invalid. Stored Hash: {block.block_hash[:8]}, Required Prefix ('0' * {block.difficulty}): '{difficulty_prefix}'"

            # 4. Check the Merkle Root: The stored merkle_root must match the root calculated from the transactions list stored in the block
            calculated_merkle_root = build_merkle_tree(block.transactions)
            if block.merkle_root != calculated_merkle_root:
                 # This indicates the transactions list within the block has been altered
                 return False, f"Block {i} Merkle root mismatch. Stored: {block.merkle_root[:8] if block.merkle_root else 'None'}, Calculated (from transactions): {calculated_merkle_root[:8] if calculated_merkle_root else 'None'}"

            # 5. Validate all transactions within the block
            # Iterate through the list of transaction dictionaries stored in the block
            for j, tx_dict in enumerate(block.transactions):
                # Skip the coinbase transaction's full validation as it's a special case with sender "0"
                # Perform basic checks for coinbase transaction
                if tx_dict.get("sender") == "0":
                     # Ensure it has the special coinbase signature
                     if tx_dict.get("signature") != "coinbase":
                           return False, f"Block {i} contains transaction {j} with sender '0' but invalid signature."
                     # You might add checks for the reward amount calculation here in a more advanced validation

                else:
                    # For standard user transactions, create a Transaction object from the dictionary
                    # The from_dict method validates presence of required keys and basic types
                    try:
                        tx = Transaction.from_dict(tx_dict)
                    except ValueError as e:
                        return False, f"Block {i} contains invalid transaction dictionary at index {j}: {e}"

                    # Validate the transaction object itself (checks signature and hash consistency)
                    if not tx.is_valid():
                        # The is_valid method logs specific reasons for failure (missing fields, bad signature, hash mismatch)
                        return False, f"Block {i} contains invalid transaction {tx_dict.get('tx_hash', 'N/A')[:8]}... at index {j}. Details logged."

        # If the loop completes without finding any issues, the chain is valid
        logger.info("Blockchain validation successful. Chain is valid.")
        return True, None

    def get_balance(self, address: str) -> float:
        """Calculates the balance for a given address by summing up transaction amounts."""
        balance = 0.0
        # Retrieve all transaction dictionaries related to the address from the DB
        # This gets both transactions sent by and received by the address
        address_txs = self.db.get_address_transactions(address)

        # Iterate through the retrieved transaction dictionaries and update the balance
        for tx_data in address_txs:
            # Add the amount if the address is the recipient
            if tx_data.get("recipient") == address:
                balance += tx_data.get("amount", 0.0) # Use .get() with default to handle potential missing key
            # Subtract the amount and fee if the address is the sender
            elif tx_data.get("sender") == address:
                 # Ensure sender is not "0" as coinbase transactions don't affect sender "0"'s balance in this way
                 if tx_data.get("sender") != "0":
                    balance -= (tx_data.get("amount", 0.0) + tx_data.get("fee", 0.0))

        return balance

    def tamper_block(self, block_idx: int, new_tx_data: Dict):
        """
        Simulates tampering with a block by modifying the first transaction's data.
        This is for demonstration purposes to show how validation detects changes.
        The tampered block is saved back to the database, overwriting the original.
        """
        # Retrieve the block from the database by index
        block = self.db.get_block_by_index_db(block_idx)

        # Check if the block exists and has transactions to tamper
        if not block or not block.transactions:
            logger.warning(f"Cannot tamper block {block_idx}: Block not found or has no transactions.")
            return

        logger.warning(f"\n!!! SIMULATING TAMPERING WITH BLOCK {block_idx} !!!")

        # Store a copy of the original hash before modification for logging
        original_hash = block.block_hash
        original_merkle_root = block.merkle_root

        # Access the first transaction's dictionary within the block's transactions list
        if block.transactions:
            first_tx = block.transactions[0]
            logger.info(f"Modifying transaction {first_tx.get('tx_hash', 'N/A')[:10]}... in block {block_idx}.")
            # Update the first transaction dictionary with the provided new data
            # This modifies the dictionary *in place* within the block's transactions list
            first_tx.update(new_tx_data)
             # Note: This simple update doesn't recalculate the transaction's individual hash or signature validity.
             # In a real scenario, changing tx data would invalidate its hash and signature.
             # This demo focuses on how changing tx data affects the block's Merkle root and hash.

        else:
            logger.warning(f"Block {block_idx} has no transactions to tamper.")
            return # Cannot tamper if no transactions

        # Recalculate the Merkle root for the block based on the modified transactions list
        # This will produce a different Merkle root if the transaction data was changed
        block.merkle_root = build_merkle_tree(block.transactions)

        # Recalculate the block hash based on the updated content (including the new Merkle root)
        # Since the nonce remains the same, this new hash will likely NOT satisfy the original difficulty requirement
        block.block_hash = block.calculate_block_hash()

        # Save the modified block back to the database (this overwrites the original block record)
        self.db.save_block(block)

        logger.warning(f"Block {block_idx} tampered.")
        logger.warning(f"  Original Block Hash: {original_hash[:12]}...")
        logger.warning(f"  New Calculated Block Hash: {block.block_hash[:12]}...")
        logger.warning(f"  Original Merkle Root: {original_merkle_root[:12] if original_merkle_root else 'None'}...")
        logger.warning(f"  New Calculated Merkle Root: {block.merkle_root[:12] if block.merkle_root else 'None'}...")
        logger.warning("Chain is now likely invalid due to hash link and/or Merkle root mismatch.")


        # Add info about the tampering event to the block creation log for the table display
        # We create a new entry to show the tampering action
        self.block_creation_log.append({
            "Index": f"{block.index} (TAMPERED)", # Indicate tampering in the index column
            "Timestamp": f"{time.time():.0f}", # Use current time for the tampering event
            "Mined By": "SIMULATED TAMPER",
            "Nonce": block.nonce, # The nonce is unchanged
            "Difficulty": block.difficulty, # The difficulty is unchanged
            "# Txs": len(block.transactions),
            "Block Hash": block.block_hash[:12]+"...",
            "Mining Time (s)": "N/A", # Not a mining event
            "Notes": f"Original Hash: {original_hash[:12]}... Tx data changed -> Merkle/Hash mismatch"
        })

    def close_db(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None # Set connection to None to indicate it's closed
            logger.info("Database connection closed.")


# ============================
# Fraud Detection Functions
# (Placeholder functions, not fully integrated with blockchain logic in this demo)
# ============================
# These functions are kept largely as provided, with minor fixes or comments for clarity.

def load_credit_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Loads credit card transaction data from a CSV file using pandas."""
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        logger.info(f"Credit data loaded successfully from {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        logger.error(f"Credit data file not found: {csv_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading credit data from {csv_path}: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> Tuple[Any, Any, Any, Any, StandardScaler]:
    """Preprocesses the data for the fraud detection model: splits into features/target, splits into train/test sets, and scales features."""
    # Separate features (X) and target variable (y)
    # 'Class' is assumed to be the target column indicating fraud (1) or legitimate (0)
    if 'Class' not in df.columns:
        logger.error("Target column 'Class' not found in the DataFrame. Cannot preprocess data.")
        raise ValueError("Target column 'Class' not found.")

    X = df.drop(columns=['Class']) # Features are all columns except 'Class'
    y = df['Class'] # Target is the 'Class' column

    # Split data into training and testing sets
    # test_size=0.3 means 30% of data goes to testing, 70% to training
    # random_state=42 ensures reproducibility of the split
    # stratify=y ensures that the proportion of fraud cases is the same in both training and testing sets, which is crucial for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize and fit the StandardScaler on the training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train) # Fit the scaler and transform the training features
    # Store the feature names from the training data after preprocessing (before scaling)
    # This is needed to ensure new data has the same columns in the correct order for scaling
    scaler.feature_names_in_ = X_train.columns.tolist() # Assuming X_train is a DataFrame

    # Transform the test features using the *same* scaler fitted on the training data
    X_test_s = scaler.transform(X_test)

    logger.info("Data preprocessed (split into train/test, scaled features).")
    # Return scaled features, target variables, and the fitted scaler object
    return X_train_s, X_test_s, y_train, y_test, scaler

def train_fraud_model(X_train: Any, y_train: Any) -> RandomForestClassifier:
    """Trains a RandomForestClassifier model for fraud detection on the provided training data."""
    logger.info("Training fraud detection model...")
    # Determine the number of CPU cores to use for training
    # Use all available cores if greater than 1, otherwise use 1
    n_jobs = multiprocessing.cpu_count() if multiprocessing.cpu_count() > 1 else 1
    logger.debug(f"Using {n_jobs} CPU cores for training.")

    # Initialize the RandomForestClassifier model
    # n_estimators: The number of trees in the forest.
    # max_depth: The maximum depth of the tree.
    # random_state: Controls the randomness of the bootstrapping and feature sampling.
    # n_jobs: Number of jobs to run in parallel for fit and predict. -1 uses all available processors.
    model = RandomForestClassifier(
        n_estimators=100, # Common starting point
        max_depth=10, # Limit depth to prevent overfitting, adjust as needed
        random_state=42, # For reproducibility
        n_jobs=n_jobs # Use available CPU cores for parallel processing
    )

    # Fit the model to the training data
    model.fit(X_train, y_train)
    logger.info("Fraud detection model training complete (RandomForestClassifier).")
    return model

def evaluate_model(model: RandomForestClassifier, X_test: Any, y_test: Any):
    """Evaluates the trained model on the test data and prints standard classification metrics."""
    logger.info("Evaluating fraud detection model...")
    # Predict the classes for the test set
    preds = model.predict(X_test)

    # Print evaluation metrics
    print("\n--- Fraud Detection Model Evaluation ---")

    # Print Confusion Matrix: A table used to evaluate the performance of a classification model
    # [[True Negatives, False Positives], [False Negatives, True Positives]]
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Print Classification Report: Text report showing the main classification metrics (Precision, Recall, F1-Score) for each class
    print("\nClassification Report:")
    # digits=4 for higher precision in the output metrics
    print(classification_report(y_test, preds, digits=4))
    print("--------------------------------------")

def detect_fraud(model: RandomForestClassifier, scaler: StandardScaler, new_tx_df: pd.DataFrame) -> Any:
    """
    Predicts fraud probability/class for new transaction data using the trained model and scaler.
    Input `new_tx_df` should be a pandas DataFrame containing the same features (columns)
    as the training data, excluding the target variable ('Class').
    Returns a numpy array of predicted classes (0 or 1).
    """
    try:
        # Ensure the columns in the new data match the columns the scaler was fitted on
        # This is crucial for consistent scaling and prediction.
        # Use feature_names_in_ attribute from the fitted scaler (if available)
        if hasattr(scaler, 'feature_names_in_'):
            expected_columns = scaler.feature_names_in_
        else:
            # Fallback: assume the new data has the same number of columns as the data used for fitting
            # This is less robust if column order or specific columns differ but count is same.
            logger.warning("Scaler does not have 'feature_names_in_'. Assuming input DataFrame columns match training data columns in order.")
            expected_columns = list(new_tx_df.columns) # Use columns from the new data as a fallback


        # Check if all expected columns are present in the new data
        if not all(col in new_tx_df.columns for col in expected_columns):
             missing_cols = [col for col in expected_columns if col not in new_tx_df.columns]
             logger.error(f"New transaction DataFrame is missing expected columns for scaling: {missing_cols}. Cannot proceed with prediction.")
             # Return a list of -1s with the same number of rows as the input DataFrame to indicate failure
             return [-1] * len(new_tx_df)

        # Ensure the new data DataFrame has the same columns and order as the training data used for the scaler
        # Select and reorder columns in the new DataFrame based on the expected columns
        new_tx_df_ordered = new_tx_df[expected_columns]

        # Scale the new transaction data using the *same* StandardScaler fitted on the training data
        # The scaler's transform method returns a numpy array
        scaled_data = scaler.transform(new_tx_df_ordered)

        # Predict the fraud class (0 for legitimate, 1 for fraud) using the trained model
        # The model's predict method expects a numpy array (which scaled_data is)
        prediction = model.predict(scaled_data)

        logger.debug(f"Fraud detection predictions: {prediction}")
        # Return the predicted class (0 or 1) for each input transaction as a numpy array
        return prediction

    except Exception as e:
        logger.error(f"Error during fraud detection prediction: {e}")
        # Return a value indicating an error occurred for each input row
        return [-1] * len(new_tx_df) # Return a list of -1s if prediction fails for each input row


def create_sample_fraud_data(num_samples: int = 5) -> pd.DataFrame:
    """
    Creates a small pandas DataFrame simulating new transaction data
    with features similar to the credit card fraud dataset for demonstration.
    This is a basic simulation and does not represent real-world fraud patterns.
    """
    # Using placeholder column names (V1 to V28, Amount, Time) similar to the typical dataset
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']

    data = {}
    for feature in feature_names:
        if feature == 'Amount':
            # Simulate varying transaction amounts, potentially with some larger values to simulate suspicious transactions
            data[feature] = [random.uniform(1.0, 500.0) if random.random() < 0.9 else random.uniform(500.0, 5000.0) for _ in range(num_samples)]
        elif feature == 'Time':
             # Simulate timestamps (arbitrary values for demo purposes)
             data[feature] = [time.time() + random.uniform(0, 10000) for _ in range(num_samples)]
        else:
            # Simulate other anonymized features (V1-V28) using random normal distribution
            data[feature] = [random.gauss(0, 1) for _ in range(num_samples)]

    # Create a pandas DataFrame from the generated data with specified feature names
    df = pd.DataFrame(data, columns=feature_names) # Explicitly set columns to ensure correct names and order
    logger.info(f"Created {num_samples} sample transactions DataFrame for fraud detection demo.")
    return df

# ============================
# Main Execution & Demo Setup
# ============================
def create_sample_blockchain_and_wallets():
    """
    Sets up a demo blockchain in memory, creates sample wallets,
    adds sample transactions, mines blocks, demonstrates validation,
    and simulates tampering.
    """
    print("\n--- Starting Blockchain Demo ---")

    # Initialize the blockchain. The db_path is effectively ignored by BlockchainDB for the in-memory demo.
    # Pass the config path as the theoretical db_path.
    blockchain = Blockchain(db_path=config.get("DATABASE", "path", fallback="sample_inmemory.db"))

    # Data structures to hold information for tabular display at the end of the demo
    wallets_info = [] # To display wallet addresses
    tx_log = [] # To log transaction creation attempts
    balance_stages = [] # To track balances at different points

    # Create sample wallets for demonstration
    alice = Wallet(); wallets_info.append({"Owner": "Alice", "Address": alice.get_address()[:12]+"..."})
    bob = Wallet(); wallets_info.append({"Owner": "Bob", "Address": bob.get_address()[:12]+"..."})
    charlie = Wallet(); wallets_info.append({"Owner": "Charlie", "Address": charlie.get_address()[:12]+"..."})
    miner = Wallet(); wallets_info.append({"Owner": "Miner", "Address": miner.get_address()[:12]+"..."}) # Wallet for the miner

    print("\n--- Sample Wallet Addresses ---")
    # Display wallet addresses using the chosen tabulate method
    print(tabulate(wallets_info, headers="keys", tablefmt="grid" if TABULATE_AVAILABLE else "pipe"))

    # Mine initial blocks to give the miner (Alice in this case) some reward coins via coinbase transactions
    logger.info("\nMining initial blocks (by Alice) to distribute reward...")
    # Mine a few blocks more than the difficulty adjustment interval to demonstrate adjustment
    blocks_to_mine_initially = blockchain.difficulty_adjustment_interval + 2
    for i in range(blocks_to_mine_initially):
        blockchain.mine_pending_transactions(alice.get_address()) # Alice acts as the miner

    # Record balances after initial mining
    balance_stages.append({
        "Stage": f"After Initial Mining ({blocks_to_mine_initially} Blocks by Alice)",
        "Alice": f"{blockchain.get_balance(alice.get_address()):.2f}",
        "Bob": f"{blockchain.get_balance(bob.get_address()):.2f}", # Bob and Charlie should have 0 balance initially
        "Charlie": f"{blockchain.get_balance(charlie.get_address()):.2f}",
        "Miner": f"{blockchain.get_balance(miner.get_address()):.2f}" # Miner's balance should reflect rewards if miner_address was used
    })

    logger.info("\nCreating and adding sample transactions to the pool...")
    # Create and sign sample transactions, then attempt to add them to the transaction pool

    # Transaction 1: Alice sends to Bob
    tx1_dict = {"sender": alice.get_address(), "recipient": bob.get_address(), "amount": 10.0, "fee": 0.1, "public_key": alice.get_public_key_string()}
    tx1 = Transaction.from_dict(tx1_dict) # Create Transaction object
    if tx1.sign(alice): # Sign the transaction
        if blockchain.add_transaction(tx1): # Add to pool (includes is_valid check)
            tx_log.append({"From": "Alice", "To": "Bob", "Amount": 10.0, "Fee": 0.1, "Hash": tx1.tx_hash[:10]+"...", "Status": "Added", "Notes": ""})
        else:
            tx_log.append({"From": "Alice", "To": "Bob", "Amount": 10.0, "Fee": 0.1, "Hash": tx1.tx_hash[:10]+"...", "Status": "Rejected", "Notes": "Invalid or Duplicate"})
    else:
         tx_log.append({"From": "Alice", "To": "Bob", "Amount": 10.0, "Fee": 0.1, "Hash": "N/A", "Status": "Rejected", "Notes": "Signing Failed"})


    # Transaction 2: Alice sends to Charlie
    tx2_dict = {"sender": alice.get_address(), "recipient": charlie.get_address(), "amount": 5.0, "fee": 0.05, "public_key": alice.get_public_key_string()}
    tx2 = Transaction.from_dict(tx2_dict)
    if tx2.sign(alice):
        if blockchain.add_transaction(tx2):
             tx_log.append({"From": "Alice", "To": "Charlie", "Amount": 5.0, "Fee": 0.05, "Hash": tx2.tx_hash[:10]+"...", "Status": "Added", "Notes": ""})
        else:
             tx_log.append({"From": "Alice", "To": "Charlie", "Amount": 5.0, "Fee": 0.05, "Hash": tx2.tx_hash[:10]+"...", "Status": "Rejected", "Notes": "Invalid or Duplicate"})
    else:
         tx_log.append({"From": "Alice", "To": "Charlie", "Amount": 5.0, "Fee": 0.05, "Hash": "N/A", "Status": "Rejected", "Notes": "Signing Failed"})


    # Transaction 3: Bob sends to Charlie (Bob likely has insufficient balance initially)
    # This transaction should be marked as invalid when added to the pool in a real system
    # Here, add_transaction checks `is_valid` which includes basic checks but not balance check against blockchain state.
    tx3_dict = {"sender": bob.get_address(), "recipient": charlie.get_address(), "amount": 3.0, "fee": 0.02, "public_key": bob.get_public_key_string()}
    tx3 = Transaction.from_dict(tx3_dict)
    if tx3.sign(bob):
        if blockchain.add_transaction(tx3):
             tx_log.append({"From": "Bob", "To": "Charlie", "Amount": 3.0, "Fee": 0.02, "Hash": tx3.tx_hash[:10]+"...", "Status": "Added", "Notes": "Added to pool (Balance check needed before mining)"}) # It will be added if valid signature/format
        else:
             tx_log.append({"From": "Bob", "To": "Charlie", "Amount": 3.0, "Fee": 0.02, "Hash": tx3.tx_hash[:10]+"...", "Status": "Rejected", "Notes": "Invalid (e.g., Signature/Format)"})
    else:
         tx_log.append({"From": "Bob", "To": "Charlie", "Amount": 3.0, "Fee": 0.02, "Hash": "N/A", "Status": "Rejected", "Notes": "Signing Failed"})


    print("\n--- Created Transactions Log ---")
    if tx_log:
        # Use the global tabulate function
        print(tabulate(tx_log, headers="keys", tablefmt="grid" if TABULATE_AVAILABLE else "pipe"))
    else:
        print("No transactions were created.")
    print(f"Current pending transactions in pool: {blockchain.transaction_pool.get_pending_count()}")


    logger.info("\nMining another block (by Miner) to include pending transactions...")
    # Mine a block. This will select pending transactions from the pool (up to max_transactions_per_block - 1)
    # and include a coinbase transaction paying the miner.
    blockchain.mine_pending_transactions(miner.get_address())

    # Record balances after mining this block
    balance_stages.append({
        "Stage": "After Mining Block (by Miner)",
        "Alice": f"{blockchain.get_balance(alice.get_address()):.2f}",
        "Bob": f"{blockchain.get_balance(bob.get_address()):.2f}",
        "Charlie": f"{blockchain.get_balance(charlie.get_address()):.2f}",
        "Miner": f"{blockchain.get_balance(miner.get_address()):.2f}"
    })

    logger.info("\nMining additional blocks to trigger difficulty adjustment demonstration...")
    # Mine more blocks to trigger the difficulty adjustment logic if the interval is reached.
    # The number of blocks needed depends on the initial blocks and the adjustment interval.
    latest_index = blockchain.db.get_latest_block().index
    blocks_mined_so_far = latest_index + 1
    # Mine enough additional blocks to reach or exceed the next adjustment point
    blocks_needed_for_next_adj = blockchain.difficulty_adjustment_interval - (blocks_mined_so_far % blockchain.difficulty_adjustment_interval)
    if blocks_needed_for_next_adj == blockchain.difficulty_adjustment_interval: # If we just completed an interval, mine a full new interval
         blocks_needed_for_next_adj = blockchain.difficulty_adjustment_interval
    # Mine one more block beyond the adjustment point to see the new difficulty in action
    blocks_to_mine_further = blocks_needed_for_next_adj + 1

    logger.info(f"Mining {blocks_to_mine_further} additional blocks (by Alice) to demonstrate difficulty adjustment.")
    for i in range(blocks_to_mine_further):
         blockchain.mine_pending_transactions(alice.get_address()) # Alice mines again


    # Record balances after final mining stage
    balance_stages.append({
        "Stage": "After Final Mining Stage (by Alice)",
        "Alice": f"{blockchain.get_balance(alice.get_address()):.2f}",
        "Bob": f"{blockchain.get_balance(bob.get_address()):.2f}",
        "Charlie": f"{blockchain.get_balance(charlie.get_address()):.2f}",
        "Miner": f"{blockchain.get_balance(miner.get_address()):.2f}"
    })


    print("\n--- Wallet Balances Over Time ---")
    if balance_stages:
        # Use the global tabulate function
        print(tabulate(balance_stages, headers="keys", tablefmt="grid" if TABULATE_AVAILABLE else "pipe"))
    else:
        print("No balance stages recorded.")

    print("\n--- Blockchain Mining Log ---")
    if blockchain.block_creation_log:
        # Use the global tabulate function
        print(tabulate(blockchain.block_creation_log, headers="keys", tablefmt="grid" if TABULATE_AVAILABLE else "pipe"))
    else:
        print("No blocks were mined.")


    # --- Validate the chain ---
    print("\n--- Validating Blockchain Integrity ---")
    is_valid, message = blockchain.is_chain_valid()
    print(f"Chain is valid: {is_valid}")
    if not is_valid:
        print(f"Validation error: {message}")
        logger.error(f"Blockchain validation failed: {message}")
    else:
        logger.info("Blockchain validation successful.")

    # --- Demo of Tampering (Optional) ---
    # Get the latest block index to select a block to tamper with
    latest_block = blockchain.db.get_latest_block()
    # Ensure there's at least one block after the genesis block to tamper
    if latest_block and latest_block.index >= 1:
        # Choose the first block after genesis (index 1) for tampering demonstration
        tamper_idx = 1
        logger.warning(f"\nAttempting to simulate tampering with block {tamper_idx}...")

        # Define new data to inject into the first transaction of the chosen block
        # Changing the amount or recipient will alter the transaction's hash,
        # which in turn alters the Merkle root of the block, and thus the block's hash.
        tamper_data = {"amount": 9999.99, "recipient": "tampered_address"} # Example: change amount and recipient

        blockchain.tamper_block(tamper_idx, tamper_data)

        # Re-validate the chain after tampering to demonstrate detection
        print(f"\n--- Validating Blockchain After Tampering Block {tamper_idx} ---")
        is_valid_after_tamper, message_after_tamper = blockchain.is_chain_valid()
        print(f"Chain valid after tampering block {tamper_idx}: {is_valid_after_tamper}")
        if not is_valid_after_tamper:
            print(f"Validation error: {message_after_tamper}")
            logger.error(f"Blockchain validation failed after tampering: {message_after_tamper}")

            # Re-evaluate balances after tampering - they might be incorrect if validation failed
            # Note: get_balance operates on the *tampered* data in the DB. If validation fails,
            # these balances are based on potentially incorrect historical data.
            print("\n--- Wallet Balances After Tampering (Based on Tampered Data) ---")
            balances_after_tamper = {
                 "Stage": f"After Tampering Block {tamper_idx}",
                 "Alice": f"{blockchain.get_balance(alice.get_address()):.2f}",
                 "Bob": f"{blockchain.get_balance(bob.get_address()):.2f}",
                 "Charlie": f"{blockchain.get_balance(charlie.get_address()):.2f}",
                 "Miner": f"{blockchain.get_balance(miner.get_address()):.2f}"
            }
            # Use the global tabulate function
            print(tabulate([balances_after_tamper], headers="keys", tablefmt="grid" if TABULATE_AVAILABLE else "pipe"))
        else:
            logger.info("Blockchain validation successful after tampering (unexpected - check tampering logic).")
            # This case should ideally not be reached if tampering was successful in invalidating the chain.

    else:
        logger.info("Not enough blocks to demonstrate tampering (need at least 2 blocks).")


    # Close the database connection
    blockchain.close_db()
    print("\n--- Blockchain Demo Finished ---")


def run_fraud_detection_demo(csv_path: str):
    """
    Runs the fraud detection pipeline: loads data, preprocesses, trains a model,
    evaluates the model, and demonstrates prediction on sample new data.
    Requires the 'creditcard.csv' file.
    """
    print("\n--- Starting Fraud Detection Demo ---")

    # Load credit card transaction data from the specified CSV file
    df = load_credit_data(csv_path)
    if df is None:
        print("Fraud detection demo skipped due to data loading failure.")
        return # Exit if data loading failed

    # Preprocess the loaded data
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    except Exception as e:
        logger.error(f"Fraud detection data preprocessing failed: {e}")
        print("Fraud detection demo skipped due to preprocessing failure.")
        return # Exit if preprocessing failed

    # Train the fraud detection model (RandomForestClassifier)
    try:
        model = train_fraud_model(X_train, y_train)
    except Exception as e:
        logger.error(f"Fraud detection model training failed: {e}")
        print("Fraud detection demo skipped due to training failure.")
        return # Exit if training failed

    # Evaluate the trained model on the test set
    try:
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        logger.error(f"Fraud detection model evaluation failed: {e}")
        # Continue if evaluation fails, as the model might still be usable for prediction
        print("Warning: Fraud detection model evaluation failed, proceeding to prediction demo.")


    # Demonstrate fraud detection on sample new transaction data
    print("\n--- Detecting Fraud on Sample New Transactions ---")
    # Create a small DataFrame with sample transaction data
    sample_new_data_df = create_sample_fraud_data(num_samples=10) # Create 10 sample transactions

    try:
        # Use the trained model and scaler to predict fraud for the sample data
        predictions = detect_fraud(model, scaler, sample_new_data_df)

        # Display the sample transactions along with their predicted fraud status
        sample_results = sample_new_data_df.copy()
        sample_results['Predicted_Fraud'] = predictions # Add prediction results as a new column

        print("Sample Transactions with Fraud Prediction:")
        # Use tabulate for console output if available, otherwise pandas display (good for notebooks)
        if TABULATE_AVAILABLE:
            # Convert DataFrame to list of lists for tabulate, preserving column order
            print(tabulate(sample_results.values.tolist(), headers=list(sample_results.columns), tablefmt="grid"))
        elif IPYTHON_AVAILABLE:
             display(sample_results) # Use pandas display in Jupyter/Colab
        else:
             # Fallback to printing the DataFrame string representation
             print(sample_results.to_string())


        # Summarize the fraud detection results
        # Count how many transactions were predicted as fraudulent (where prediction is 1)
        fraud_count = sum(p == 1 for p in predictions if p != -1) # Exclude -1 if prediction failed
        total_samples = len(predictions)
        print(f"\nDetected {fraud_count} potentially fraudulent transaction(s) out of {total_samples} sample transactions.")
        if any(p == -1 for p in predictions):
             print("Note: Some predictions failed due to an error.")


    except Exception as e:
        logger.error(f"Fraud detection on sample data failed: {e}")
        print("\nFraud detection on sample data could not be completed due to an error.")


def main():
    """Main function to parse arguments and run the combined blockchain and fraud detection demo."""
    # Ensure logging is configured before anything else
    setup_logging()
    logger.info("--- Starting Combined Demo ---")

    # Setup argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Blockchain + Fraud Detection Demo")
    # Argument to specify the path to the credit card fraud dataset CSV file
    parser.add_argument("--cc_data", default="creditcard.csv",
                         help="Path to creditcard.csv dataset for fraud detection (default: creditcard.csv)")
    # Add flags to optionally skip demos
    parser.add_argument("--no_blockchain_demo", action="store_true", help="Skip the blockchain demo.")
    parser.add_argument("--no_fraud_demo", action="store_true", help="Skip the fraud detection demo.")


    # Check if running in a notebook environment (Colab/Jupyter) to handle arguments appropriately
    # In notebooks, sys.argv might be just the script name, causing issues with parse_args()
    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        # In a notebook, we might not have command-line arguments in sys.argv.
        # We can create a default Namespace object with default argument values.
        # If needed, you could add a mechanism to pass args in a notebook cell.
        args = argparse.Namespace(cc_data="creditcard.csv", no_blockchain_demo=False, no_fraud_demo=False)
        logger.info("Detected notebook environment. Using default arguments.")
        # If you want to allow passing args in a notebook cell, you could parse sys.argv
        # and handle the case where it's empty or contains only the script name.
        # Example: args = parser.parse_args([] if len(sys.argv) <= 1 else sys.argv[1:])
    else:
        # Standard command-line parsing
        args = parser.parse_args()
        logger.info("Detected command-line environment. Parsing arguments.")


    # --- Run Blockchain Demo (if not skipped) ---
    if not args.no_blockchain_demo:
        try:
            create_sample_blockchain_and_wallets()
        except Exception as e:
            logger.critical(f"Blockchain demo failed critically: {e}")
            print("\n!!! Blockchain demo encountered a critical error and stopped. !!!")
    else:
        print("\n--- Skipping Blockchain Demo as requested ---")
        logger.info("Skipping blockchain demo.")


    # --- Run Fraud Detection Demo (if not skipped) ---
    if not args.no_fraud_demo:
        # Check if the creditcard.csv file exists at the specified path before attempting to run the demo
        credit_data_path = args.cc_data
        if os.path.isfile(credit_data_path):
            try:
                run_fraud_detection_demo(credit_data_path)
            except Exception as e:
                logger.critical(f"Fraud detection demo failed critically: {e}")
                print("\n!!! Fraud detection demo encountered a critical error and stopped. !!!")
        else:
            # If the CSV file is not found, inform the user and skip the demo
            logger.warning(f"Fraud detection data file not found at {credit_data_path}. Skipping fraud detection demo.")
            print(f"\n--- Skipping Fraud Detection Demo ---")
            print(f"The required file '{credit_data_path}' was not found.")
            print("Please download the 'Credit Card Fraud Detection' dataset (e.g., from Kaggle) and place 'creditcard.csv' at this path or provide the correct path via --cc_data argument.")
    else:
        print("\n--- Skipping Fraud Detection Demo as requested ---")
        logger.info("Skipping fraud detection demo.")


    print("\n--- Combined Demo Finished ---")


# --- Install necessary libraries if not already installed ---
# These checks and installations should run when the script is imported or executed.
# Use quiet mode (-q) for cleaner output in environments like Colab/Jupyter.

# Check and install tabulate
try:
    import tabulate
    TABULATE_AVAILABLE = True # Ensure this flag is set correctly after potential install
except ImportError:
    print("Installing 'tabulate' library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabulate"])
    try:
        import tabulate
        TABULATE_AVAILABLE = True
        print("Successfully installed 'tabulate'.")
    except ImportError:
        TABULATE_AVAILABLE = False
        print("Failed to install 'tabulate'. Manual table formatting will be used.")


# Check and install cryptography
try:
    import cryptography
except ImportError:
    print("Installing 'cryptography' library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cryptography"])
    try:
        import cryptography
        print("Successfully installed 'cryptography'.")
    except ImportError:
        print("Failed to install 'cryptography'. Cryptography features may not work.")

# Check and install pandas and scikit-learn
try:
    import pandas
    import sklearn
except ImportError:
    print("Installing 'pandas' and 'scikit-learn' libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "scikit-learn"])
    try:
        import pandas
        import sklearn
        print("Successfully installed 'pandas' and 'scikit-learn'.")
    except ImportError:
        print("Failed to install 'pandas' and 'scikit-learn'. Fraud detection demo may not work.")


# --- Script Entry Point ---
# This __name__ == "__main__" block allows the code to be imported as a module
# without automatically running the demo. The demo runs only when the script
# is executed directly.
if __name__ == "__main__":
    main() # Call the main function to start the demo\
    