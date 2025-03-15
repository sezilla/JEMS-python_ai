from sshtunnel import SSHTunnelForwarder
import logging
import os
import time
import paramiko

logger = logging.getLogger("ssh_tunnel")

class SSHTunnel:
    def __init__(self, ssh_host, ssh_user, ssh_pkey, remote_host, remote_port, local_port=13306):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_pkey = ssh_pkey
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.tunnel = None
        self.tunnel_is_up = False
        self.max_retries = 3
        
    def start(self):
        if self.tunnel_is_up:
            logger.debug("Tunnel is already running")
            return

        retries = 0
        while retries < self.max_retries:
            try:
                # Normalize path - replace Windows backslashes with forward slashes
                normalized_key_path = self.ssh_pkey.replace('\\', '/')
                
                # Check if file exists and is readable
                if not os.path.isfile(normalized_key_path):
                    logger.error(f"SSH private key not found at: {normalized_key_path}")
                    raise Exception(f"SSH private key not found at: {normalized_key_path}")
                
                logger.info(f"Setting up SSH tunnel to {self.ssh_host} using key: {normalized_key_path}")
                
                # Set logging level for paramiko to reduce noise
                logging.getLogger("paramiko").setLevel(logging.WARNING)
                
                # Create tunnel with improved configuration
                self.tunnel = SSHTunnelForwarder(
                    (self.ssh_host, 22),
                    ssh_username=self.ssh_user,
                    ssh_pkey=normalized_key_path,
                    remote_bind_address=(self.remote_host, self.remote_port),
                    local_bind_address=('127.0.0.1', self.local_port),
                    set_keepalive=10,  # Send keepalive every 10 seconds
                    ssh_config_file=None,  # Don't use system SSH config files
                    compression=True,  # Enable compression for faster data transfer
                    mute_exceptions=False  # Don't mute exceptions to better diagnose issues
                )
                
                logger.info("Starting SSH tunnel...")
                self.tunnel.start()
                
                # Allow more time for tunnel establishment
                time.sleep(2)
                
                if self.tunnel.is_active:
                    self.tunnel_is_up = True
                    local_address = f"{self.tunnel.local_bind_host}:{self.tunnel.local_bind_port}"
                    logger.info(f"SSH tunnel established: {local_address} → {self.ssh_host} → {self.remote_host}:{self.remote_port}")
                    break  # Successfully established tunnel, exit retry loop
                else:
                    logger.error("Tunnel started but not active, retrying...")
                    self._cleanup_tunnel()
                    retries += 1
                    time.sleep(1)  # Wait before retrying
            except Exception as e:
                logger.error(f"Error starting SSH tunnel (attempt {retries+1}/{self.max_retries}): {str(e)}")
                self._cleanup_tunnel()
                retries += 1
                if retries >= self.max_retries:
                    self.tunnel_is_up = False
                    raise Exception(f"Failed to establish SSH tunnel after {self.max_retries} attempts: {str(e)}")
                time.sleep(2)  # Wait before retrying

    def _cleanup_tunnel(self):
        """Clean up tunnel resources in case of failure"""
        if self.tunnel:
            try:
                self.tunnel.stop()
            except:
                pass
            self.tunnel = None
        self.tunnel_is_up = False

    def stop(self):
        if self.tunnel_is_up and self.tunnel:
            try:
                self.tunnel.stop()
                self.tunnel_is_up = False
                logger.info("SSH tunnel stopped")
            except Exception as e:
                logger.error(f"Error stopping SSH tunnel: {str(e)}")

    def check_is_alive(self):
        """Verify if the tunnel is still active and reconnect if not"""
        if not self.tunnel or not self.tunnel.is_active:
            logger.warning("SSH tunnel is not active, attempting to restart")
            self._cleanup_tunnel()
            self.start()
            return self.tunnel_is_up
        return True