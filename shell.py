import pty
import subprocess
import os

UNIQUE_SEPARATOR = '__rehydrate_the_masses__'


class Shell:
    def __init__(self, cwd = None, shell_args = None):
        if shell_args is None:
            shell_args = ['/bin/bash']
        self.master_fd, self.slave_fd = pty.openpty()
        kwargs = {}
        kwargs['cwd'] = cwd
        self.process = subprocess.Popen(
            shell_args + ['--noprofile', '--norc', '-i'],
            stdin=subprocess.PIPE,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            text=True,
            close_fds=True,
            env={
                "PS1": UNIQUE_SEPARATOR,
            },
            **kwargs
        )
        self.master = os.fdopen(self.master_fd, 'rb')

        os.close(self.slave_fd)

        self.read_until_prompt()

    def close(self):
        self.master.close()
        self.process.terminate()

    def read_until_prompt(self):
        chunks = []
        while True:
            chunk = os.read(self.master_fd, 1024).decode('utf-8')
            if chunk.strip().endswith(UNIQUE_SEPARATOR):
                break
            chunks.append(chunk)
        return ''.join(chunks)

    def run_command(self, command):
        if self.master is None:
            raise RuntimeError("Shell should be called using the with statement")

        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

        # read one chunk at a time, until we see the UNIQUE_SEPARATOR
        return self.read_until_prompt()