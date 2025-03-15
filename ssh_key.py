#!/usr/bin/env python3
"""
SSH Key Generator for Development Environment
This script generates an SSH key pair for local development and testing.
"""

import os
import subprocess
import platform
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_key_directory():
    """Create a directory for storing the SSH key"""
    # Create a 'secret' directory in the current working directory
    secret_dir = os.path.join(os.getcwd(), "secret")
    if not os.path.exists(secret_dir):
        os.makedirs(secret_dir)
        logger.info(f"Created directory: {secret_dir}")
    return secret_dir

def generate_ssh_key(key_path):
    """Generate an SSH key pair using ssh-keygen"""
    try:
        # Check if ssh-keygen is available
        if shutil.which("ssh-keygen") is None:
            logger.error("ssh-keygen command not found. Please install OpenSSH.")
            return False
        
        # Generate key with empty passphrase
        subprocess.run([
            "ssh-keygen", 
            "-t", "rsa",
            "-b", "2048",
            "-f", key_path,
            "-N", "",  # Empty passphrase
            "-C", f"dev-key-{platform.node()}"  # Comment with hostname
        ], check=True)
        
        logger.info(f"SSH key pair generated at: {key_path}")
        logger.info(f"Public key: {key_path}.pub")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate SSH key: {e}")
        return False

def update_env_file(key_path):
    """Update the .env file with the SSH key path"""
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        # Normalize path for the current platform and .env file
        normalized_path = key_path.replace("\\", "/")
        
        # Find and update the SSH_PRIVATE_KEY line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("SSH_PRIVATE_KEY="):
                lines[i] = f"SSH_PRIVATE_KEY={normalized_path}\n"
                updated = True
                break
        
        # If not found, add it
        if not updated:
            lines.append(f"SSH_PRIVATE_KEY={normalized_path}\n")
        
        # Write the updated .env file
        with open(env_path, "w") as f:
            f.writelines(lines)
        
        logger.info(f"Updated .env file with SSH key path: {normalized_path}")
    else:
        logger.warning(f".env file not found at {env_path}")
        # Create a new .env file with the SSH key path
        with open(env_path, "w") as f:
            f.write(f"SSH_PRIVATE_KEY={normalized_path}\n")
        logger.info(f"Created new .env file with SSH key path: {normalized_path}")

def display_key_fingerprint(key_path):
    """Display the fingerprint of the generated SSH key"""
    try:
        result = subprocess.run([
            "ssh-keygen", 
            "-l", 
            "-f", f"{key_path}.pub"
        ], capture_output=True, text=True, check=True)
        
        fingerprint = result.stdout.strip()
        logger.info(f"Key fingerprint: {fingerprint}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get key fingerprint: {e}")

def main():
    """Main function to generate SSH key and update environment"""
    logger.info("Starting SSH key generation process...")
    
    # Create directory for storing the key
    secret_dir = create_key_directory()
    
    # Define the key path
    key_path = os.path.join(secret_dir, "dev_id_rsa")
    
    # Check if key already exists
    if os.path.exists(key_path):
        logger.warning(f"SSH key already exists at {key_path}")
        overwrite = input("Overwrite existing key? (y/n): ").lower() == 'y'
        if not overwrite:
            logger.info("Using existing SSH key")
            display_key_fingerprint(key_path)
            update_env_file(key_path)
            return
    
    # Generate new SSH key
    if generate_ssh_key(key_path):
        display_key_fingerprint(key_path)
        update_env_file(key_path)
        
        # Display the public key for easy copying
        try:
            with open(f"{key_path}.pub", "r") as f:
                public_key = f.read().strip()
            logger.info("Public key (for adding to authorized_keys):")
            print(f"\n{public_key}\n")
        except IOError as e:
            logger.error(f"Failed to read public key: {e}")
    else:
        logger.error("SSH key generation failed")

if __name__ == "__main__":
    main()