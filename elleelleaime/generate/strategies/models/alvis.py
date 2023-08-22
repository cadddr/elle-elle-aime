from generate.strategies.strategy import PatchGenerationStrategy
from dotenv import load_dotenv
from typing import Any

import os
import uuid
import paramiko
import json
import threading


MODELS_DICT = {
    "facebook/incoder-6B": {
        "inference_script": "incoder.py",
    },
    "facebook/incoder-1B": {
        "inference_script": "incoder.py",
    },
}


class AlvisHFModels(PatchGenerationStrategy):
    __SETUP_LOCK: threading.Lock = threading.Lock()
    __SETUP_FLAG: bool = False

    def __init__(self, model: str, **kwargs) -> None:
        assert (
            model in MODELS_DICT.keys()
        ), f"Model {model} not supported by AlvisHFModels"
        self.model = model
        # Compute settings
        self.gpu_type = kwargs.get("gpu_type", "A40")
        self.gpu_number = kwargs.get("gpu_number", "1")
        self.job_time = kwargs.get("job_time", "00:10:00")
        # Generation settings
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        self.num_return_sequences = kwargs.get("num_return_sequences", 10)
        self.temperature = kwargs.get("temperature", 0.0)
        # Alvis settings
        load_dotenv()
        self.hostname = os.getenv("ALVIS_HOSTNAME", "alvis")
        self.username = os.getenv("ALVIS_USERNAME")
        self.password = os.getenv("ALVIS_PASSWORD")
        self.project = os.getenv("ALVIS_PROJECT")

    def __setup_remote_env(self, ssh: paramiko.SSHClient):
        with self.__SETUP_LOCK:
            if not self.__SETUP_FLAG:
                # Copy the required files to the remote cluster
                with ssh.open_sftp() as sftp:
                    # mkdir elleelleaime if it does not exist
                    command = (
                        f"mkdir -p /cephyr/users/{self.username}/Alvis/elleelleaime/"
                    )
                    stdin, stdout, stderr = ssh.exec_command(command)
                    stdout.channel.recv_exit_status()
                    stderr.channel.recv_exit_status()

                    # Write the setup_env.sh scripts to the remote server
                    with sftp.open(
                        f"/cephyr/users/{self.username}/Alvis/elleelleaime/setup_env.sh",
                        "w+",
                    ) as f:
                        with open(
                            "./generate/resources/scripts/setup_env.sh", "r"
                        ) as lf:
                            script = lf.read()
                            f.write(script)

                    # Write the load_modules.sh script to the remote server
                    with sftp.open(
                        f"/cephyr/users/{self.username}/Alvis/elleelleaime/load_modules.sh",
                        "w+",
                    ) as f:
                        with open(
                            "./generate/resources/scripts/load_modules.sh", "r"
                        ) as lf:
                            script = lf.read()
                            f.write(script)

                    # Write the requirements.txt file to the remote server
                    with sftp.open(
                        f"/cephyr/users/{self.username}/Alvis/elleelleaime/requirements.txt",
                        "w+",
                    ) as f:
                        with open(
                            "./generate/resources/scripts/requirements.txt", "r"
                        ) as lf:
                            script = lf.read()
                            f.write(script)

                # Setup the remote environment
                command = f"cd /cephyr/users/{self.username}/Alvis/elleelleaime/ && \
                            source load_modules.sh && source setup_env.sh && \
                            pip install -r requirements.txt && wait"
                stdin, stdout, stderr = ssh.exec_command(command)
                stdout.channel.recv_exit_status()
                stderr.channel.recv_exit_status()
                self.__SETUP_FLAG = True

    def _generate_impl(self, prompt: str) -> Any:
        # Generate unique id
        unique_id = str(uuid.uuid4())

        # Connect to remote cluster using paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.hostname, username=self.username, password=self.password
        )

        # Setup the remote environment if needed
        self.__setup_remote_env(ssh)

        # Create scripts and test data on the remote cluster
        self.__create_scripts(ssh, prompt, unique_id)

        # Submit job to Slurm on the remote cluster and wait for its conclusion
        self.__submit_job(ssh, unique_id)

        # Retrieve the results
        results = self.__retrieve_results(ssh, unique_id)

        # Cleanup the remote cluster
        self.__cleanup_remote(ssh, unique_id)

        # Close the SSH connection
        ssh.close()

        return results

    def __create_scripts(self, ssh: paramiko.SSHClient, prompt: str, unique_id: str):
        with ssh.open_sftp() as sftp:
            # Initialize the directory structure
            sftp.mkdir(
                f"/cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/"
            )

            # Write the test samples file to a file on the remote cluster
            with sftp.open(
                f"/cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/inputs.txt",
                "w+",
            ) as f:
                f.write(prompt)

            # Write the HuggingFace script to a file on the remote cluster
            with sftp.open(
                f"/cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/inference.py",
                "w+",
            ) as f:
                with open(
                    f"./generate/resources/scripts/huggingface/{MODELS_DICT[self.model]['inference_script']}",
                    "r",
                ) as lf:
                    script = lf.read().format(
                        model=self.model,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=self.num_return_sequences,
                        temperature=self.temperature,
                        unique_id=unique_id,
                    )
                    f.write(script)

            # Write the jobscript to a file on the remote cluster
            with sftp.open(
                f"/cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/jobscript",
                "w+",
            ) as f:
                with open("./generate/resources/scripts/jobscript", "r") as lf:
                    script = lf.read().format(
                        job_name="elleelleaime",
                        gpu_type=self.gpu_type,
                        gpu_number=self.gpu_number,
                        job_time=self.job_time,
                        project=self.project,
                        script=f"/cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/inference.py",
                    )
                    f.write(script)

    def __submit_job(self, ssh: paramiko.SSHClient, unique_id: str):
        # Submit job to Slurm
        command = f"cd /cephyr/users/{self.username}/Alvis/elleelleaime/ && source setup_env.sh && \
                        cd elleelleaime-{unique_id}/ && sbatch --wait jobscript && wait"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.recv_exit_status()
        stderr.channel.recv_exit_status()

    def __retrieve_results(self, ssh: paramiko.SSHClient, unique_id: str):
        # Retrieve the results from the remote cluster
        with ssh.open_sftp() as sftp:
            sftp.get(
                f"/cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/predictions.txt",
                f"{unique_id}_predictions.txt",
            )

        # Read the output file and return the results as a string
        with open(f"{unique_id}_predictions.txt", "r") as f:
            results = json.loads(f.read())

        # Remove the output file
        os.remove(f"{unique_id}_predictions.txt")

        return results

    def __cleanup_remote(self, ssh: paramiko.SSHClient, unique_id: str):
        # Cleanup the remote cluster
        command = f"rm -rf /cephyr/users/{self.username}/Alvis/elleelleaime/elleelleaime-{unique_id}/"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.recv_exit_status()
        stderr.channel.recv_exit_status()