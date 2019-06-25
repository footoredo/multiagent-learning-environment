from env.base_env import BaseEnv
from gym import spaces
import random
import numpy as np
from typing import List, Callable, Union, Tuple, NoReturn, Dict
from env.pokemon_data import BattleTypeChart


class Type(object):
    def __init__(self, name: str, i: int, psn: int = 0, tox: int = 0, prankster: int = 0,
                 par: int = 0, brn: int = 0, hail: int = 0, frz: int = 0, trapped: int = 0, powder: int = 0):
        self.name = name
        self.i = i
        self.psn = psn
        self.tox = tox
        self.prankster = prankster
        self.par = par
        self.brn = brn
        self.hail = hail
        self.frz = frz
        self.trapped = trapped
        self.powder = powder


def json_to_type(name, i, json):
    return Type(name, i, **json)


TYPES = [json_to_type(name, i, json) for i, name, json in enumerate(BattleTypeChart.items())]
TYPES_DICT = dict([(t.name, t) for t in TYPES])


def get_type(name: str):
    return TYPES_DICT[name]


def get_chart_row(damage_taken: Dict[str]):
    row = [None] * len(TYPES)
    for t_name, v in damage_taken.items():
        row[TYPES_DICT[t_name].i] = v
    return row


DAMAGE_TAKEN_CHART = [get_chart_row(json["damageTaken"]) for _, json in BattleTypeChart.items()]


STAT_NAMES = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]


class Pokemon(object):
    def __init__(self, name: str, base_values: List[int], move_set: List[int]):
        self.name = name
        assert len(base_values) == 6
        self.base_values = base_values
        self.move_set = move_set

    @staticmethod
    def _get_value(base_value: int, iv: int, ev: int, level: int, nature: int) -> int:
        value = (2 * base_value + iv + ev // 4) * level // 100 + 5
        if nature > 0:
            value = value * 11 // 10
        elif nature < 0:
            value = value * 9 // 10
        # print(value, nature)
        return value

    @staticmethod
    def _get_hp_value(base_value: int, iv: int, ev: int, level: int) -> int:
        return (2 * base_value + iv + ev // 4) * level // 100 + level + 10

    @property
    def hp(self) -> int:
        return self.base_values[0]

    @property
    def attack(self) -> int:
        return self.base_values[1]

    @property
    def defense(self) -> int:
        return self.base_values[2]

    @property
    def sp_attack(self) -> int:
        return self.base_values[3]

    @property
    def sp_defense(self) -> int:
        return self.base_values[4]

    @property
    def speed(self) -> int:
        return self.base_values[5]

    def get_hp_value(self, iv: int, ev: int, level: int) -> int:
        return self._get_hp_value(self.hp, iv, ev, level)

    def get_value(self, i: int, iv: int, ev: int, level: int, nature: int) -> int:
        return self._get_value(self.base_values[i], iv, ev, level, nature)


class PokemonInstance(object):
    def __init__(self, pokemon: Pokemon, moves: List[int], level: int, ivs: List[int], evs: List[int],
                 nature: Tuple[int, int], nickname=None):
        self.name = nickname or pokemon.name
        self.pokemon = pokemon
        self.moves = moves

        for move in moves:
            assert move in self.pokemon.move_set

        self.level = level
        self.hp = pokemon.get_hp_value(ivs[0], evs[0], level)
        # print(self.hp)
        natures = [0 for _ in range(6)]
        natures[nature[0]] += 1
        natures[nature[1]] -= 1
        self.attack, self.defense, self.sp_attack, self.sp_defense, self.speed = \
            [pokemon.get_value(i, ivs[i], evs[i], level, natures[i]) for i in range(1, 6)]

        self.current_hp = None
        self.protecting = None
        self.protect_success_rate = None
        self.protect_count = None
        self.stat_stages = None

        self.display()

    def display(self):
        print(self.name)
        print(STAT_NAMES[0], self.hp)
        print(STAT_NAMES[1], self.attack)
        print(STAT_NAMES[2], self.defense)
        print(STAT_NAMES[3], self.sp_attack)
        print(STAT_NAMES[4], self.sp_defense)
        print(STAT_NAMES[5], self.speed)

    def debug_hp(self):
        return "{}/{} HP".format(self.current_hp, self.hp)

    def debug_protect_count(self):
        return "protect count: {}".format(self.protect_count)

    def debug_stat_stages(self):
        ret = []
        for i in range(6):
            if self.stat_stages[i] != 0:
                ret.append("{:+d} {}".format(self.stat_stages[i], STAT_NAMES[i]))
        return ' '.join(ret)

    def debug(self):
        print("{}: {}".format(self.name,
                              ', '.join([self.debug_hp(),
                                         self.debug_protect_count(),
                                         self.debug_stat_stages()])))

    def reset(self):
        self.current_hp = self.hp
        self.protecting = False
        self.protect_success_rate = 1
        self.protect_count = 0
        self.stat_stages = [0] * 6

    def turn_reset(self):
        if self.protecting:
            self.protect_success_rate *= 3
            self.protect_count += 1
            self.protecting = False
        else:
            self.protect_success_rate = 1
            self.protect_count = 0

    def lose_hp(self, amount):
        self.current_hp -= amount

    def protect(self):
        self.protecting = True

    def stat_boost(self, i, amount):
        self.stat_stages[i] += amount
        if self.stat_stages[i] > 6:
            self.stat_stages[i] = 6

    def stat_drop(self, i, amount):
        self.stat_stages[i] -= amount
        if self.stat_stages[i] < -6:
            self.stat_stages[i] = -6

    @property
    def n_moves(self):
        return len(self.moves)

    def get_move(self, i):
        return self.moves[i]

    @property
    def is_dead(self):
        return self.current_hp <= 0


class RNG(object):
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def judge(self, n: int, m: int):
        return self._randint(0, m - 1) < n

    def _randint(self, a: int, b: int):
        return self.random.randint(a, b)

    def _shuffle(self, l: list):
        return self.random.shuffle(l)

    @staticmethod
    def apply_stat_stage(base: int, stage: int):
        if stage > 0:
            ret = base * (stage + 2) // 2
        else:
            ret = base * 2 // (2 - stage)
        return ret

    def calc_damage(self, level: int, attack: int, defense: int, attack_stat_stage: int, defense_stat_stage: int,
                    power: int, accuracy: int) -> int:
        if self.judge(100 - accuracy, 100):
            return 0

        ct = self.judge(1, 24)

        if ct:
            if attack_stat_stage < 0:
                attack_stat_stage = 0
            if defense_stat_stage > 0:
                defense_stat_stage = 0

        attack = self.apply_stat_stage(attack, attack_stat_stage)
        defense = self.apply_stat_stage(defense, defense_stat_stage)

        damage = ((2 * level) // 5 + 2) * power * attack // defense // 50 + 2
        if ct:
            damage = damage * 3 // 2
        damage = damage * self._randint(85, 100) // 100
        return damage

    def determine_order(self, p: list):
        op = list(zip(range(len(p)), p))
        self._shuffle(op)
        op.sort(key=lambda x: x[1], reverse=True)
        # print(list(*op)[0])
        return list(list(zip(*op))[0])


class GlobalStats(object):
    def __init__(self):
        pass

    def reset(self):
        pass


class MoveEnv(object):
    def __init__(self, pokemons: List[PokemonInstance], global_stats: GlobalStats, side: int, rng: RNG):
        self.pokemons = pokemons
        self.global_stats = global_stats
        self.side = side
        self.rng = rng


class AttackMoveEnv(object):
    def __init__(self, move_env: MoveEnv):
        self.attacker = move_env.pokemons[0] if move_env.side == 0 else move_env.pokemons[1]
        self.defender = move_env.pokemons[1] if move_env.side == 0 else move_env.pokemons[0]
        self.global_stats = move_env.global_stats
        self.side = move_env.side
        self.rng = move_env.rng


class SelfMoveEnv(object):
    def __init__(self, move_env: MoveEnv):
        self.actor = move_env.pokemons[move_env.side]
        self.global_stats = move_env.global_stats
        self.side = move_env.side
        self.rng = move_env.rng


class Move(object):
    def __init__(self, name: str, act: Callable[[MoveEnv], NoReturn], priority: int = 0):
        self.name = name
        self.act = act
        self.priority = priority

    @staticmethod
    def attack_move(power_func: Callable[[AttackMoveEnv], int], use_sp_attack: bool, use_sp_defense: bool,
                    accuracy: int) -> Callable[[MoveEnv], NoReturn]:
        def act(move_env: MoveEnv):
            attack_move_env = AttackMoveEnv(move_env)
            attacker, defender = attack_move_env.attacker, attack_move_env.defender

            if defender.protecting:
                return

            attack = attacker.sp_attack if use_sp_attack else attacker.attack
            defense = defender.sp_defense if use_sp_defense else defender.defense
            attack_stat_stage = attacker.stat_stages[3] if use_sp_attack else attacker.stat_stages[1]
            defense_stat_stage = defender.stat_stages[4] if use_sp_defense else defender.stat_stages[2]
            power = power_func(attack_move_env)

            damage = move_env.rng.calc_damage(level=attacker.level, attack=attack, defense=defense,
                                              attack_stat_stage=attack_stat_stage, defense_stat_stage=defense_stat_stage,
                                              power=power, accuracy=accuracy)

            if damage > defender.current_hp:
                damage = defender.current_hp

            defender.lose_hp(damage)
        return act

    @staticmethod
    def ph_attack_move(power_func: Callable[[AttackMoveEnv], int], accuracy: int) -> Callable[[MoveEnv], NoReturn]:
        return Move.attack_move(power_func, use_sp_attack=False, use_sp_defense=False, accuracy=accuracy)

    @staticmethod
    def sp_attack_move(power_func: Callable[[AttackMoveEnv], int], accuracy: int) -> Callable[[MoveEnv], NoReturn]:
        return Move.attack_move(power_func, use_sp_attack=True, use_sp_defense=True, accuracy=accuracy)

    @staticmethod
    def constant_power(power: int) -> Callable[[AttackMoveEnv], int]:
        def power_func(_: AttackMoveEnv) -> int:
            return power
        return power_func

    @staticmethod
    def hp_ratio_power(base_power: int) -> Callable[[AttackMoveEnv], int]:
        def power_func(env: AttackMoveEnv) -> int:
            return base_power * env.attacker.current_hp // env.attacker.hp
        return power_func

    @staticmethod
    def protect_act(move_env: MoveEnv):
        actor = move_env.pokemons[move_env.side]
        if move_env.rng.judge(1, actor.protect_success_rate):
            actor.protect()

    @staticmethod
    def stat_boost_move(i: int, amount: int, is_self: bool):
        def act(move_env: MoveEnv):
            attack_move_env = AttackMoveEnv(move_env)
            attacker, defender = attack_move_env.attacker, attack_move_env.defender
            if is_self:
                attacker.stat_boost(i, amount)
            else:
                defender.stat_boost(i, amount)
        return act

    @staticmethod
    def stat_drop_move(i: int, amount: int, is_self: bool):
        def act(move_env: MoveEnv):
            attack_move_env = AttackMoveEnv(move_env)
            attacker, defender = attack_move_env.attacker, attack_move_env.defender
            if is_self:
                attacker.stat_drop(i, amount)
            else:
                defender.stat_drop(i, amount)

        return act


class MultipleMove(Move):
    def __init__(self, name: str, acts: List[Callable[[MoveEnv], NoReturn]], priority: int = 0):
        def act(move_env: MoveEnv):
            for a in acts:
                a(move_env)

        super().__init__(name, act, priority)


MOVES = [
    Move("Protect", Move.protect_act, priority=4),
    Move("Tackle", Move.ph_attack_move(Move.constant_power(40), accuracy=100)),
    Move("Pound", Move.ph_attack_move(Move.constant_power(40), accuracy=100)),
    Move("Water Gun", Move.sp_attack_move(Move.constant_power(40), accuracy=100)),
    Move("Growl", Move.stat_drop_move(1, 1, False)),
    Move("Sword Dance", Move.stat_boost_move(1, 2, True)),
    Move("Water Spout", Move.sp_attack_move(Move.hp_ratio_power(150), accuracy=100)),
    Move("Origin Pulse", Move.sp_attack_move(Move.constant_power(110), accuracy=85)),
    Move("Precipice Blades", Move.ph_attack_move(Move.constant_power(120), accuracy=85)),
    Move("Fire Punch", Move.ph_attack_move(Move.constant_power(75), accuracy=100))
]

MOVES_DICT = dict([(move.name, i) for i, move in enumerate(MOVES)])


def get_move_number(name: str) -> int:
    return MOVES_DICT[name]


def get_move(name: str) -> Move:
    return MOVES[get_move_number(name)]


POKEMONS = [
    Pokemon("Primal Kyogre", base_values=[100, 150, 90, 180, 160, 90],
            move_set=[
                MOVES_DICT["Water Spout"],
                MOVES_DICT["Origin Pulse"],
                MOVES_DICT["Protect"]
            ]),
    Pokemon("Primal Groudon", base_values=[100, 180, 160, 150, 90, 90],
            move_set=[
                MOVES_DICT["Precipice Blades"],
                MOVES_DICT["Fire Punch"],
                MOVES_DICT["Sword Dance"],
                MOVES_DICT["Protect"]
            ]),
    Pokemon("Caterpie", base_values=[45, 30, 35, 20, 20, 45],
            move_set=[
                MOVES_DICT["Protect"]
            ]),
    Pokemon("Popplio", base_values=[50, 54, 54, 66, 56, 40],
            move_set=[
                MOVES_DICT["Water Gun"],
                MOVES_DICT["Pound"]
            ]),
    Pokemon("Rowlet", base_values=[68, 55, 55, 50, 50, 42],
            move_set=[
                MOVES_DICT["Tackle"],
                MOVES_DICT["Growl"]
            ])
]

POKEMONS_DICT = dict([(pokemon.name, i) for i, pokemon in enumerate(POKEMONS)])


def get_pokemon_number(name: str) -> int:
    return POKEMONS_DICT[name]


def get_pokemon(name: str) -> Pokemon:
    return POKEMONS[get_pokemon_number(name)]


class Game(object):
    def __init__(self, pokemons: List[PokemonInstance], seed=None):
        assert len(pokemons) == 2
        self.pokemons = pokemons
        self.n_pokemons = len(pokemons)
        self.seed = seed
        self.rng = None
        self.global_stats = GlobalStats()
        self.is_over = None
        self.winner = None
        self.debug = None

    def reset(self, debug=False):
        self.rng = RNG(self.seed)
        for pokemon in self.pokemons:
            pokemon.reset()
        self.global_stats.reset()
        self.is_over = False
        self.winner = None
        self.debug = debug

    def step(self, actions):
        assert len(actions) == self.n_pokemons
        moves = [MOVES[self.pokemons[i].get_move(actions[i])] for i in range(self.n_pokemons)]
        order_p = [(moves[i].priority, self.pokemons[i].speed) for i in range(self.n_pokemons)]
        order = self.rng.determine_order(order_p)

        for i in order:
            if self.debug:
                print("{} used {}.".format(self.pokemons[i].name, moves[i].name))

            move = moves[i]
            move.act(MoveEnv(self.pokemons, self.global_stats, i, self.rng))
            for pokemon in self.pokemons:
                if pokemon.is_dead:
                    self.is_over = True
                    self.winner = i
                    break
            if self.is_over:
                break

        for i in order:
            self.pokemons[i].turn_reset()
            if self.debug:
                self.pokemons[i].debug()
            if self.pokemons[i].is_dead:
                self.is_over = True
                self.winner = 0 if i == 1 else 1
                break


class PokemonEnv(BaseEnv):
    def __init__(self, a: PokemonInstance, b: PokemonInstance, seed=None):
        self.game = Game([a, b], seed)

        ob_len = 7
        ob_spaces = [spaces.Box(low=0., high=1., shape=[ob_len * 2])] * 2
        ac_spaces = [spaces.Discrete(a.n_moves), spaces.Discrete(b.n_moves)]

        super().__init__(num_agents=2,
                         observation_spaces=ob_spaces,
                         action_spaces=ac_spaces)

    @staticmethod
    def _get_ob_pokemon(pokemon: PokemonInstance) -> np.array:
        ob = np.zeros(shape=7, dtype=np.float32)
        ob[0] = pokemon.current_hp / pokemon.hp
        ob[1] = 1. / pokemon.protect_success_rate
        ob[2:7] = np.array(pokemon.stat_stages[1:]) / 12. + .5
        return ob

    def _get_ob(self):
        return [np.concatenate((self._get_ob_pokemon(self.game.pokemons[0]),
                                self._get_ob_pokemon(self.game.pokemons[1])))] * 2

    def reset(self, debug=False):
        self.game.reset(debug)
        return self._get_ob()

    def step(self, actions):
        self.game.step(actions)
        a_rew = 0.
        b_rew = 0.
        if self.game.is_over:
            if self.game.winner == 0:
                a_rew = 1.
                b_rew = -1.
            else:
                a_rew = -1.
                b_rew = 1.

        # print(self._get_ob())

        return self._get_ob(), [a_rew, b_rew], [None, None], self.game.is_over

    # def _move(self, i, j):
    #     attacker = self.a if i == 0 else self.b
    #     move = attacker.get_move(j)
    #     defender = self.b if i == 0 else self.a
    #     damage = self.rng.deal_damage(attacker, move, defender)
    #     defender.lose_hp(damage)
    #     # print(i, j, damage)
    #     return defender.is_dead, damage
    #
    # def step(self, actions):
    #     order = [self.a, self.b]
    #     self.rng.determine_order(order)
    #     a_win, b_win = False, False
    #     a_damage, b_damage = 0, 0
    #     if order[0] == self.a:
    #         a_win, a_damage = self._move(0, actions[0])
    #         if not a_win:
    #             b_win, b_damage = self._move(1, actions[1])
    #     else:
    #         # assert False
    #         b_win, b_damage = self._move(1, actions[1])
    #         if not b_win:
    #             a_win, a_damage = self._move(0, actions[0])
    #
    #     a_rew = 0.
    #     b_rew = 0.
    #     if a_win:
    #         a_rew = 1.
    #         b_rew = -1.
    #     elif b_win:
    #         a_rew = -1.
    #         b_rew = 1.
    #     return self._get_ob(), [a_rew, b_rew], [a_damage, b_damage], a_win or b_win

    def get_ob_namers(self):
        def ob_namer(ob):
            return ob.astype(int).tolist()
        return [ob_namer] * 2
