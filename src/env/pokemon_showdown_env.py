from env.base_env import BaseEnv
from gym import spaces
import subprocess
import json

SHOWDOWN_PATH = "/home/footoredo/playground/Pokemon-Showdown/pokemon-showdown"


class SimulatorCommunicator(object):
    def __init__(self):
        self.simulator = subprocess.Popen([SHOWDOWN_PATH, 'simulate-battle'],
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)

    def _write(self, s):
        self.simulator.stdin.write(str.encode(s))

    def _flush(self):
        self.simulator.stdin.flush()

    def send(self, *args):
        self._write('>')
        for part in args:
            if type(part) == str:
                self._write(part)
            else:
                self._write(json.dumps(part))
            self._write(' ')
        self._write('\n')
        self._flush()

    def read(self):
        return bytes.decode(self.simulator.stdout.readline()).strip()

    def terminate(self):
        if self.simulator.poll() is None:
            self.simulator.terminate()

    def __del__(self):
        self.terminate()


class BattleSimulator(object):
    def __init__(self, seed=None, p1_seed=None, p2_seed=None):
        self.comm = SimulatorCommunicator()
        options = {
            "formatid": "testletsgobattle",
            "seed": seed
        }
        self._send('start', options)
        self._send('player', 'p1', {"name": "Alice", "seed": p1_seed})
        self._send('player', 'p2', {"name": "Bob", "seed": p2_seed})

    def _send(self, *args):
        self.comm.send(*args)

    def _read(self):
        return self.comm.read()

    def _read_message(self):
        return json.loads(self._read().split('|')[1:]

    def _read_side_update(self):
        self._read()   # sideupdate
        player = self._read()   # player
        message = self._read_message()  # message
        return json.loads(message[1])

    @staticmethod
    def _convert_pokemon_message(message):
        hp, conditions = message["condition"].split(' ')
        rem_hp, full_hp = list(map(int, hp.split('/')))
        return {
            "hp": rem_hp / full_hp,
            "active": message["active"],
            "brn": "brn" in conditions,
            "par": "par" in conditions,
            "slp": "slp" in conditions,
            "frz": "frz" in conditions,
            "psn": "psn" in conditions,
            "tox": "tox" in conditions
        }

    @staticmethod
    def _convert_side_message(message):
        return [BattleSimulator._convert_pokemon_message(msg) for msg in message["side"]["pokemon"]]

if __name__ == "__main__":
    battle = BattleSimulator(p1_seed=[0, 0, 0, 0], p2_seed=[33, 44, 66, 22])
    print(json.dumps(battle.read_side_update()))
