import os.path

import random
import paramiko
import time
import subprocess
import hashlib

from hdt.util import *
from hdt.util import get_hdt_eval_path, get_hdt_eval_file


def upload_with_password(remote_dir, local_path, local_file, target_user, target_host):
    def calculate_md5(file_path):
        """Calculate MD5 hash of a local file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    password = os.getenv("SSH_PASSWORD")
    remote_path = os.path.join(remote_dir, local_file)

    # Retry configuration
    max_retries = 5
    base_delay = 2  # seconds
    max_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            ssh = paramiko.SSHClient()
            # Automatically add the server's host key (like saying 'yes' to the scp prompt)
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            print(f"Connecting to {target_host}... (attempt {attempt + 1}/{max_retries})")

            # Add timeout and connection parameters
            ssh.connect(
                hostname=target_host,
                username=target_user,
                password=password,
                timeout=30,  # Connection timeout
                banner_timeout=30,  # Banner read timeout
                auth_timeout=30  # Authentication timeout
            )

            sftp = ssh.open_sftp()
            try:
                sftp.chdir(remote_dir)  # Try to enter the directory
            except IOError:
                print(f"Creating remote directory {remote_dir}...")
                sftp.mkdir(remote_dir)  # Create it if it doesn't exist
                sftp.chdir(remote_dir)

            # Calculate local file MD5
            local_md5 = calculate_md5(local_path)
            print(f"Local file MD5: {local_md5}")

            # Check if remote file exists and compare MD5
            should_upload = True
            try:
                # Try to stat the remote file
                sftp.stat(remote_path)
                print(f"Remote file exists, checking MD5...")

                # Calculate remote file MD5
                stdin, stdout, stderr = ssh.exec_command(f"md5sum {remote_path}")
                remote_md5_output = stdout.read().decode().strip()
                remote_md5 = remote_md5_output.split()[0] if remote_md5_output else None

                if remote_md5:
                    print(f"Remote file MD5: {remote_md5}")
                    if local_md5 == remote_md5:
                        print(f"File {local_file} already exists with same MD5. Skipping upload.")
                        should_upload = False
                    else:
                        print(f"File {local_file} exists but MD5 differs. Will upload.")
                else:
                    print(f"Could not calculate remote MD5. Will upload.")
            except IOError:
                print(f"Remote file does not exist. Will upload.")

            if should_upload:
                print(f"Uploading {local_file} to {remote_path}...")
                sftp.put(local_path, remote_path)
                print("Upload successful!")

            sftp.close()
            ssh.close()

            # Success - exit retry loop
            return

        except paramiko.SSHException as e:
            error_msg = str(e)
            if "Error reading SSH protocol banner" in error_msg or "Connection reset by peer" in error_msg:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"Connection error: {e}")
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached. Final error: {e}")
                    raise
            else:
                # Different SSH error - don't retry
                print(f"SSH error (not retrying): {e}")
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached. Giving up.")


def monitor_and_parse(file_path):
    print(f"Monitoring for file: {file_path}")

    while not os.path.exists(file_path):
        print("File not found. Checking again in 60 seconds...")
        time.sleep(60)

    print("File detected!")

    # Reading the file content (which contains the URL)
    try:
        with open(file_path, 'r') as file:
            url_variable = file.read().strip()

        print(f"Successfully parsed URL: {url_variable}")
        return url_variable

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


def check_and_get_superprocess_log_url(exp, model_type, allow_simulation, period: bool = False):
    model_type_str = model_type.value if hasattr(model_type, "value") else str(model_type)
    url_file = get_hdt_eval_path(exp, model_type_str, "superprocess_url.txt", allow_simulation=allow_simulation)
    if period:
        return monitor_and_parse(url_file)
    return os.path.exists(url_file)


def retrieve_superprocess_subprocesses(log_url, model_type, exp, allow_simulation):
    index_file = get_hdt_eval_file(exp, model_type, "index.txt", allow_simulation)
    index_path = os.path.dirname(index_file)
    if not os.path.exists(index_file):
        cwd = os.getcwd()
        os.chdir(index_path)
        subprocess.run(["cpee-logging-xes-yaml", "copy", log_url])
        os.chdir(cwd)
    try:
        subprocesses, t = extract_sub_from_index(index_path)
        return subprocesses, t
    except FileNotFoundError:
        print(f"File not found: {index_file}. Likely, because the cpee-logging-xes yaml failed for {exp} {log_url}.")
        return None, None


