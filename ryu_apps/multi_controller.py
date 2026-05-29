# ryu_apps/multi_controller.py
import subprocess
import time
import os
import logging

LOG = logging.getLogger(__name__)
CTRL_PORTS = [6653, 6654, 6655, 6656, 6657]
N_CONTROLLERS = 5


class MultiControllerManager:
    """Starts/stops 5 ryu-manager processes on ports 6653-6657."""

    def __init__(self, app_path='ryu_apps/ctrl_app.py', log_dir='/tmp/ryu_logs'):
        self.app_path = app_path
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.procs = {}        # port -> Popen
        self._log_fds = {}     # port -> file handle (keep open)

    def start_all(self):
        for i, port in enumerate(CTRL_PORTS):
            logpath = f'{self.log_dir}/ctrl_{i}.log'
            logfile = open(logpath, 'w')
            self._log_fds[port] = logfile
            proc = subprocess.Popen(
                [
                    '/home/maksim/dqn_env/bin/ryu-manager', self.app_path,
                    '--ofp-tcp-listen-port', str(port),
                    # --observe-links intentionally OMITTED: causes LLDP flood after reassignment
                    '--log-file', logpath,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.procs[port] = proc
            LOG.info(f'Started Ryu ctrl {i} on port {port}, pid={proc.pid}')
        time.sleep(3)

    def stop_all(self):
        for port, proc in self.procs.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            LOG.info(f'Stopped Ryu on port {port}')
        self.procs.clear()
        for fd in self._log_fds.values():
            try:
                fd.close()
            except Exception:
                pass
        self._log_fds.clear()

    def is_running(self, ctrl_idx):
        port = CTRL_PORTS[ctrl_idx]
        proc = self.procs.get(port)
        return proc is not None and proc.poll() is None

    def restart_ctrl(self, ctrl_idx):
        port = CTRL_PORTS[ctrl_idx]
        if port in self.procs:
            try:
                self.procs[port].kill()
            except Exception:
                pass
        logpath = f'{self.log_dir}/ctrl_{ctrl_idx}_restart.log'
        logfile = open(logpath, 'w')
        self._log_fds[port] = logfile
        proc = subprocess.Popen(
            [
                '/home/maksim/dqn_env/bin/ryu-manager', self.app_path,
                '--ofp-tcp-listen-port', str(port),
                '--log-file', logpath,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.procs[port] = proc
        time.sleep(2)
        LOG.info(f'Restarted Ryu ctrl {ctrl_idx} on port {port}, pid={proc.pid}')
